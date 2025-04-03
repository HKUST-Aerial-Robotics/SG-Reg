import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import KPConvFPN, KPConvShape, SEModule
from model.backbone import encode_batch_scenes_points, encode_batch_scenes_instances
from model.gnn import GraphNeuralNetwork, NodesInitLayer
from model.match.match import MatchAssignment, SinkhornMatch
from model.match.learnable_sinkhorn import LearnableLogOptimalTransport
from model.loss.loss import FineMatchingLoss, InstanceMatchingLoss, contrastive_loss_fn
from model.loss.eval import Evaluator, eval_instance_match, eval_instance_match_new
from model.registration.local_global_registration import LocalGlobalRegistration

from model.utils.tictoc import TicToc

class SGNet(nn.Module):
    default_config = {
        "bert_dim": 768,
        "gat_heads": 4
    }
    
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.backbone = KPConvFPN(
            conf.backbone.input_dim,
            conf.backbone.output_dim,
            conf.backbone.init_dim,
            conf.backbone.kernel_size,
            conf.backbone.init_radius,
            conf.backbone.init_sigma,
            conf.backbone.group_norm
        )        
        
        self.shape_backbone = KPConvShape(
            2 * conf.backbone.init_dim * (conf.shape_encoder.input_from_stages+1),
            conf.shape_encoder.output_dim,
            conf.shape_encoder.kernel_size,
            conf.shape_encoder.init_radius,
            conf.shape_encoder.init_sigma,
            conf.shape_encoder.group_norm,
            conf.shape_encoder.point_limit,
            conf.fine_matching.num_points_per_instance
        )
        
        # Init layers: semantic, bbox and MLP layers
        self.node_int_layer = NodesInitLayer(conf.scenegraph,
                                             conf.shape_encoder.output_dim,
                                             conf.dataset.online_bert)

        # GAT layers
        self.spatial_gnn = GraphNeuralNetwork(self.conf.scenegraph) 
        self.se_layer = SEModule(self.conf.scenegraph.node_dim+
                                 self.conf.scenegraph.semantic_dim)
        
        # Scene graph match layers
        match_feat_dim = [self.conf.scenegraph.node_dim, self.conf.scenegraph.node_dim]
        if 'ltriplet' in self.conf.scenegraph.gnn.layers \
            and self.conf.scenegraph.gnn.triplet_mlp=='concat':
            match_feat_dim[-1] += self.conf.scenegraph.node_dim
        
        match_feat_dim.append(self.conf.shape_encoder.output_dim
                              +match_feat_dim[-1])
        
        self.node_matching_layers = nn.ModuleDict() # [init_node_match_layer, coarse_match_layer, dense_match_layer]
        for i in self.conf.instance_matching.match_layers:
            self.node_matching_layers[str(i)] = MatchAssignment(
                match_feat_dim[i],
                self.conf.instance_matching.min_score,
                self.conf.instance_matching.topk,
                self.conf.instance_matching.multiply_matchability)
        
        if 'sinkhorn' in self.conf.instance_matching:
            for i in self.conf.instance_matching.match_layers:
                self.sinkhorn_match_layers[str(i)] = SinkhornMatch(
                    match_feat_dim[i],
                    self.conf.instance_matching.sinkhorn_iterations,
                    self.conf.instance_matching.topk,
                    self.conf.instance_matching.min_score)

        # Point match layer
        self.optimal_transport = LearnableLogOptimalTransport(
            self.conf.fine_matching.num_sinkhorn_iterations)
        
    
    def split_feats_dict(self, 
                         instance_feats_dict:dict,
                         batch_size:int,
                         batch_points:dict,
                         ref_graph_batch:torch.Tensor,
                         src_graph_batch:torch.Tensor):
        '''
        Split the instance_feats_dict into ref_dict and src_dict.
        '''
        
        ref_instances_batch = [torch.tensor(0).to('cuda')]
        src_instances_batch = [torch.tensor(0).to('cuda')]
        ref_instance_f_points = []
        src_instance_f_points = []
        ref_instance_f_feats = []
        src_instance_f_feats = []
        ref_instance_d_feats = []
        src_instance_d_feats = []
        ref_instance_shapes = []
        src_instance_shapes = []
        
        # ref_graph_batch = batch_graph_dict['ref_graph']['batch']
        # src_graph_batch = batch_graph_dict['src_graph']['batch']
        K = self.conf.fine_matching.num_points_per_instance
        for scene_id in torch.arange(batch_size):
            data_dict = batch_points[scene_id]
            points_f = data_dict['points'][1].detach() # (P+Q,3)        
            feats_batch = instance_feats_dict['feats_batch']
            insts_batch = instance_feats_dict['instances_batch']
            feats_f = instance_feats_dict['feats_f'][
                feats_batch[scene_id]:feats_batch[scene_id+1]]
            assert feats_f.shape[0] == data_dict['lengths'][1].sum().item()

            #
            padded_points_f = torch.cat([points_f,torch.zeros_like(points_f[:1])],dim=0)    
            padded_feats_f = torch.cat([feats_f,torch.zeros_like(feats_f[:1])],dim=0)

            # Instance-wise points and features for fine matching
            instances_knn_indices = instance_feats_dict['instances_f_indices_match'][
                insts_batch[scene_id]:insts_batch[scene_id+1]
            ] # (m+n,K)
            assert instances_knn_indices.shape[1] == K
            instances_knn_points = padded_points_f[instances_knn_indices] # (m+n,K,3)
            instances_knn_feats = padded_feats_f[instances_knn_indices] # (m+n,K,C)
            
            ref_instance_number = ref_graph_batch[scene_id+1]-ref_graph_batch[scene_id] # m
            src_instance_number = src_graph_batch[scene_id+1]-src_graph_batch[scene_id] # n
            assert instances_knn_indices.shape[0] == ref_instance_number + src_instance_number
            
            #
            ref_instances_batch.append(ref_instances_batch[-1]+ref_instance_number)
            src_instances_batch.append(src_instances_batch[-1]+src_instance_number)
            ref_instance_f_points.append(instances_knn_points[:ref_instance_number])
            src_instance_f_points.append(instances_knn_points[ref_instance_number:])
            ref_instance_f_feats.append(instances_knn_feats[:ref_instance_number])
            src_instance_f_feats.append(instances_knn_feats[ref_instance_number:])            
            
            # Shape embeddings
            instance_shapes = instance_feats_dict['instances_shape'][
                insts_batch[scene_id]:insts_batch[scene_id+1]]  
            ref_instance_shapes.append(instance_shapes[:ref_instance_number])
            src_instance_shapes.append(instance_shapes[ref_instance_number:]) 
            
            # Decoded point features                        
            if 'feats_f_decoded' in instance_feats_dict:          
                feats_decode = instance_feats_dict['feats_f_decoded'][
                    feats_batch[scene_id]:feats_batch[scene_id+1]]
                padded_feats_d = torch.cat([feats_decode,torch.zeros_like(feats_decode[:1])],dim=0)      
                instances_knn_d_feats = padded_feats_d[instances_knn_indices] # (Mi+Ni,K,C)      
                ref_instance_d_feats.append(instances_knn_d_feats[:ref_instance_number])
                src_instance_d_feats.append(instances_knn_d_feats[ref_instance_number:])
        
        # collate
        ref_instances_batch = torch.stack(ref_instances_batch,dim=0) # (M,)
        src_instances_batch = torch.stack(src_instances_batch,dim=0) # (N,)
        ref_instance_f_points = torch.cat(ref_instance_f_points,dim=0) # (M,K,3)
        src_instance_f_points = torch.cat(src_instance_f_points,dim=0) # (N,K,3)
        ref_instance_f_feats = torch.cat(ref_instance_f_feats,dim=0) # (M,K,C)
        src_instance_f_feats = torch.cat(src_instance_f_feats,dim=0) # (N,K,C)
        ref_instance_shapes = torch.cat(ref_instance_shapes,dim=0) # (M,C)
        src_instance_shapes = torch.cat(src_instance_shapes,dim=0) # (N,C)        
        if 'feats_f_decoded' in instance_feats_dict:
            ref_instance_d_feats = torch.cat(ref_instance_d_feats,dim=0) # (M,K,C)
            src_instance_d_feats = torch.cat(src_instance_d_feats,dim=0) # (N,K,C)
        else:
            # decoded instance features can be ignored unless it is in pre-trainning stage.
            ref_instance_d_feats = None
            src_instance_d_feats = None
        
        return {'instances_batch':ref_instances_batch,
                'instances_f_points':ref_instance_f_points,
                'instances_f_feats':ref_instance_f_feats,
                'instances_d_feats':ref_instance_d_feats,
                'instances_shapes':ref_instance_shapes},\
                {'instances_batch':src_instances_batch,
                 'instances_f_points':src_instance_f_points,
                 'instances_f_feats':src_instance_f_feats,
                 'instances_d_feats':src_instance_d_feats,
                 'instances_shapes':src_instance_shapes}
            
    def forward(self, data_dict): 
        ''' Encode all nodes in both graphs.'''  
        torch.cuda.memory.reset_peak_memory_stats()
        torch.cuda.memory.reset_max_memory_cached()
        torch.cuda.memory.reset_max_memory_allocated()
        torch.cuda.empty_cache()
        output_dict = {}
        time_list = []
        tictoc = TicToc()
        SAVE_HIDDEN_FEATS = False # [X1, shapes, labels]
        
        # 1. Shape encoder
        if self.conf.scenegraph.fuse_shape or \
            self.conf.loss.shape_contrast_weight>0.0 or \
            self.conf.train.registration_in_train:
            # Gather fine points and features in instances
            instances_f_feats_dict,t_points = encode_batch_scenes_points(self.backbone,
                                                                data_dict)
            # print('points encoder time: {:.1f}ms, {:.1f}ms'.format(1000*t_points,
            #                                                        1000*tictoc.toc()))
            time_list.append(1000*t_points) # points

            # Shape embeddings
            instances_f_feats_dict, t_shapes = encode_batch_scenes_instances(
                                        self.shape_backbone,
                                        data_dict,
                                        instances_f_feats_dict, 
                                        self.conf.loss.shape_contrast_weight>0.0)
            time_list.append(1000*t_shapes) # shapes
            
            # Split concat dict into ref_dict and src_dict
            ref_instance_knn_dict, src_instance_knn_dict = self.split_feats_dict(instances_f_feats_dict,
                                                                                 data_dict['batch_size'],
                                                                                 data_dict['batch_points'],
                                                                                 data_dict['ref_graph']['batch'],
                                                                                 data_dict['src_graph']['batch'])
            
            output_dict['ref_instances_points_batch'] = ref_instance_knn_dict['instances_batch']
            output_dict['src_instances_points_batch'] = src_instance_knn_dict['instances_batch']
            output_dict['ref_instance_points'] = ref_instance_knn_dict['instances_f_points']
            output_dict['src_instance_points'] = src_instance_knn_dict['instances_f_points']
            output_dict['ref_instance_shapes'] = ref_instance_knn_dict['instances_shapes']
            output_dict['src_instance_shapes'] = src_instance_knn_dict['instances_shapes']    
            if instances_f_feats_dict['invalid_instance_exist']:
                output_dict['invalid_instance_exist'] = True
        else:
            ref_instance_knn_dict = {'instances_shapes':None}
            src_instance_knn_dict = {'instances_shapes':None}
            time_list.append(1000*tictoc.toc()) # points
            time_list.append(1000*tictoc.toc()) # shapes
        
        # 2. Init node features
        tictoc.tic()
        x_src0 = self.node_int_layer(data_dict['src_graph'],
                    src_instance_knn_dict['instances_shapes']) # (m,feat_dim)
        x_ref0 = self.node_int_layer(data_dict['ref_graph'],
                    ref_instance_knn_dict['instances_shapes']) # (n,feat_dim)
        time_list.append(1000*tictoc.toc()) # init nodes
        
        assert torch.isnan(x_src0).sum()==0 and \
            torch.isnan(x_ref0).sum()==0, 'NaN in node features'

        # 3. Encode scene graph by triplet-boosted GAT
        x_src, x_ref, t_gnn = self.spatial_gnn(x_src0,x_ref0,data_dict)
        time_list.append(1000*t_gnn) # gnn
        
        # 4. Fuse node features from multiple modalities
        tictoc.tic()
        if self.conf.scenegraph.fuse_shape and \
            self.conf.scenegraph.fuse_stage=='late': # fuse shape embeddings
            #todo: check in onnx it is normalize correctly.
            src_shape_embeddings = src_instance_knn_dict['instances_shapes']
            ref_shape_embeddings = ref_instance_knn_dict['instances_shapes']
            feat_src = torch.cat([x_src,src_shape_embeddings],dim=1)
            feat_ref = torch.cat([x_ref,ref_shape_embeddings],dim=1)
            l=2
        else:
            feat_src, feat_ref = x_src, x_ref
            l=1
        
        if SAVE_HIDDEN_FEATS:
            output_dict['x_src1'] = x_src.clone()
            output_dict['x_ref1'] = x_ref.clone()
            output_dict['f_src']  = src_shape_embeddings.clone()
            output_dict['f_ref']  = ref_shape_embeddings.clone()
        
        if self.conf.scenegraph.se_layer: #todo
            assert False, 'SE layer is abandoned'
            feat_src = self.se_layer(feat_src)
        time_list.append(1000*tictoc.toc()) # fuse
              
        # 5. Coarse scene graph matching
        output_dict['node_matching'] = self.match_node_layer(
            feat_src,feat_ref,data_dict,l)
        time_list.append(1000*tictoc.toc()) # match nodes
        
        # 6. Dense point matching
        if self.conf.train.registration_in_train and \
            output_dict['node_matching']['pred_nodes'].shape[0]>0: # match points
            t0 = time.time()
            output_dict['points_matching'] = self.match_point_layer(
                                        output_dict['node_matching']['pred_nodes'],
                                        data_dict,
                                        ref_instance_knn_dict,
                                        src_instance_knn_dict)
            t_point_match = time.time() - t0
            output_dict['point_matching_time'] = 1000*t_point_match
        time_list.append(1000*tictoc.toc()) # match points
        
        # 7.(Optional) only used in pre-train the shape encoder
        if self.conf.loss.fine_loss_weight>0.0: 
            output_dict['gt_points_matching_dict'] = self.match_point_layer(
                                        None,
                                        data_dict, 
                                        ref_instance_knn_dict,
                                        src_instance_knn_dict)
                
        output_dict['time_list'] = time_list
        output_dict['memory_list'] = [torch.cuda.memory_allocated()/ 1024 / 1024,
                                      torch.cuda.memory_reserved()/ 1024 / 1024,
                                      torch.cuda.max_memory_reserved()/ 1024 / 1024] 
        
        # torch.cuda.memory._dump_snapshot('sgnet_memory.pickle')
        return output_dict

    def match_node_layer(self,x_src,x_ref,batch_graph_pair,l):
        ''' Graph match semantic nodes.
        Return dict:
        - output_dict[pred_nodes]: (M,3),[batch_id,src_idx,tar_idx], 
                                 node idex are local index
        - output_dict[pred_scores]: (M,), matching scores
        - output_dict[logmax_scores]: list of logmax scores
        '''
        
        batch_pred = [] #(M,3),[scan_id,src_idx,tar_idx], node idex are local index
        batch_pred_scores = []
        batch_logscores = []
        
        for scan_id in torch.arange(batch_graph_pair['batch_size']):
            nodes_mask_src = batch_graph_pair['src_graph']['scene_mask']==scan_id
            nodes_mask_ref = batch_graph_pair['ref_graph']['scene_mask']==scan_id
            
            _, logscores, k_assignment = self.node_matching_layers[str(l)](
                        x_src[nodes_mask_src,:].unsqueeze(0), 
                        x_ref[nodes_mask_ref,:].unsqueeze(0))
            matches = k_assignment.nonzero().detach() # (m,2),[src_idx,tar_idx]
            matches_scores = torch.exp(logscores.clone().detach())
            matches_scores = matches_scores[matches[:,0],matches[:,1]] # (m,)
            assert matches_scores.shape[0] == matches.shape[0]
  
            batch_pred.append(torch.cat(
                [torch.tensor([scan_id]).repeat(matches.shape[0],1).to('cuda'),matches],
                dim=1)) # (m,3)
            batch_pred_scores.append(matches_scores)
            batch_logscores.append(logscores)

        assert len(batch_logscores) == batch_graph_pair['batch_size']
        
        with torch.no_grad():
            batch_pred = torch.cat(batch_pred,dim=0) # (M,3)
            batch_pred_scores = torch.cat(batch_pred_scores,dim=0) # (M,)
            
            batch_output_dict = {'pred_nodes':batch_pred.detach(),
                                 'pred_scores':batch_pred_scores.detach(),
                                 'logmax_scores':batch_logscores # list
                                 } 
        return batch_output_dict

    def match_point_layer(self,
                            pred_nodes,
                            data_dict,
                            ref_instance_knn_dict,
                            src_instance_knn_dict):
        '''
        Input,
        - pred_nodes: (M,3),[scan_id,src_idx,tar_idx]
        Output dict,
        - output[matching_ptr]: (M,), matching ptr
        - output[matching_scores]: (M,K+1,K+1), matching scores
        - output[ref_matching_points]: (M,K,3), matching points in ref
        - output[src_matching_points]: (M,K,3), matching points in src
        '''
        batch_match_ptr = [] # (M,)
        batch_scores_matrix = [] 
        batch_sample_pts_src = []
        batch_sample_pts_ref = []
        IGNORE_LABELS = self.conf.fine_matching.ignore_semantics 

        for scene_id in torch.arange(data_dict['batch_size']):
            # Extract stacked features and points
            ref_b = ref_instance_knn_dict['instances_batch']
            src_b = src_instance_knn_dict['instances_batch']
            points_ref = ref_instance_knn_dict['instances_f_points'][
                ref_b[scene_id]:ref_b[scene_id+1]] # (a,T,3)
            points_src = src_instance_knn_dict['instances_f_points'][
                src_b[scene_id]:src_b[scene_id+1]] # (b,T,3)
            src_labels = data_dict['src_graph']['labels'][
                data_dict['src_graph']['batch'][scene_id]
                :data_dict['src_graph']['batch'][scene_id+1]] # list
            ref_labels = data_dict['ref_graph']['labels'][
                data_dict['ref_graph']['batch'][scene_id]
                :data_dict['ref_graph']['batch'][scene_id+1]]
            
            feats_ref = ref_instance_knn_dict['instances_f_feats'][
                ref_b[scene_id]:ref_b[scene_id+1]] # (a,T,C)
            feats_src = src_instance_knn_dict['instances_f_feats'][
                src_b[scene_id]:src_b[scene_id+1]] # (b,T,C) 
            if pred_nodes is None: # guided by ground-truth node matches
                gt_matches = data_dict['instance_matches'][scene_id] 
                node_corres_indices_ref = gt_matches[:,1] # (m',)
                node_corres_indices_src = gt_matches[:,0] # (m',)
            else: # guided by predicted node matches
                cur_scene_mask = pred_nodes[:,0] == scene_id
                node_corres_indices_ref = pred_nodes[cur_scene_mask,2] # (m,)
                node_corres_indices_src = pred_nodes[cur_scene_mask,1] # (m,)
            
            # 
            if len(IGNORE_LABELS)>0:
                pred_ref_labels = [ref_labels[idx] for idx in 
                                   node_corres_indices_ref.detach().cpu().numpy()]
                pred_src_labels = [src_labels[idx] for idx in 
                                   node_corres_indices_src.detach().cpu().numpy()]
                pred_ref_nodes_valid = [False if label in IGNORE_LABELS 
                                            else True for label in pred_ref_labels]
                pred_src_nodes_valid = [False if label in IGNORE_LABELS 
                                            else True for label in pred_src_labels]
                valid = torch.tensor([pred_ref_nodes_valid and pred_src_nodes_valid]).to('cuda').squeeze()
                if valid.sum()<1:continue
                # if valid.sum()<node_corres_indices_src.shape[0]:
                #     print('ignore floors or carpets during points matching')
                node_corres_indices_ref = node_corres_indices_ref[valid]
                node_corres_indices_src = node_corres_indices_src[valid]
            node_corres_indices_ref = node_corres_indices_ref.squeeze()
            node_corres_indices_src = node_corres_indices_src.squeeze()
            
            # Extract instance-wise points and features
            sample_feats_ref = feats_ref[node_corres_indices_ref] # (m,Kp,C)
            sample_feats_src = feats_src[node_corres_indices_src] # (m,Kp,C)
            sample_pts_ref = points_ref[node_corres_indices_ref] # (m,Kp,3)
            sample_pts_src = points_src[node_corres_indices_src] # (m,Kp,3)
            
            if False:
                print('Skip using node correspondeces. Brutal match the points.')
                print('M:{}, Kp:{}'.format(sample_pts_ref.shape[0],
                                           sample_pts_ref.shape[1]))    
                sample_feats_ref = feats_ref.view(-1, feats_ref.shape[-1]) # (m*Kp,C)
                sample_feats_src = feats_src.view(-1, feats_src.shape[-1])
                sample_pts_ref = points_ref.view(-1, points_ref.shape[-1]) # (m*Kp,3)
                sample_pts_src = points_src.view(-1, points_src.shape[-1])
            
            if sample_feats_ref.ndim==2:
                sample_feats_ref = sample_feats_ref.unsqueeze(0)
                sample_feats_src = sample_feats_src.unsqueeze(0)
                sample_pts_ref = sample_pts_ref.unsqueeze(0)
                sample_pts_src = sample_pts_src.unsqueeze(0)
            
            # optimal transport
            score_matrix = torch.einsum('bmc,bnc->bmn',
                                           sample_feats_ref,
                                           sample_feats_src) # (m,Kp,Kp)
            score_matrix = score_matrix/ sample_feats_ref.shape[-1] ** 0.5
            score_matrix = self.optimal_transport(score_matrix) # (m,K+1,K+1)  
            assert score_matrix.ndim==3, 'matching scores should be 3D'
            
            # store          
            batch_sample_pts_ref.append(sample_pts_ref)
            batch_sample_pts_src.append(sample_pts_src)
            batch_match_ptr.append(scene_id.clone().unsqueeze(0).
                                   repeat(sample_pts_ref.shape[0],1).to('cuda'))
            batch_scores_matrix.append(score_matrix)

        if len(batch_scores_matrix)<1:
            return None
        batch_match_ptr = torch.cat(batch_match_ptr,dim=0) # (M,1)
        batch_scores_matrix = torch.cat(batch_scores_matrix,dim=0) # (M,Kp+1,Kp+1)
        batch_sample_pts_ref = torch.cat(batch_sample_pts_ref,dim=0) # (M,Kp,3)
        batch_sample_pts_src = torch.cat(batch_sample_pts_src,dim=0) # (M,Kp,3)
        
        assert batch_scores_matrix.ndim==3, 'matching scores should be 3D'
        if not self.conf.fine_matching.use_dustbin:
            batch_scores_matrix = batch_scores_matrix[:,:-1,:-1]
        
        return {'matching_ptr':batch_match_ptr.squeeze(),
                'matching_scores':batch_scores_matrix,
                'ref_matching_points':batch_sample_pts_ref,
                'src_matching_points':batch_sample_pts_src}
        
def SGNetDecorator(conf,test=False):
    evaluator = Evaluator(conf)
    cfg = conf
    instance_matching_fn = InstanceMatchingLoss(conf.loss)
    point_matching_loss = FineMatchingLoss(conf.loss.positive_point_radius)

    fine_registration = LocalGlobalRegistration(conf.fine_matching.topk,
                                                conf.fine_matching.acceptance_radius,
                                                conf.fine_matching.max_instance_selection,
                                                conf.fine_matching.mutual,
                                                conf.fine_matching.confidence_threshold,
                                                conf.fine_matching.use_dustbin)

    def model_fn(data_dict:dict,
                 model:nn.Module,
                 epoch:int,
                 train:bool):
        t0 = time.time()
        output_dict = model(data_dict)
        conf = model.conf
        
        # 1. Loss 
        loss_in_dict = {'node_matching':output_dict['node_matching']}
        if 'src_instance_shapes' in output_dict: 
            loss_in_dict['ref_instance_shapes'] = output_dict['ref_instance_shapes']
            loss_in_dict['src_instance_shapes'] = output_dict['src_instance_shapes']
        if 'gt_points_matching_dict' in output_dict:
            loss_in_dict['gt_points_matching_dict'] = output_dict['gt_points_matching_dict']
        
        if 'invalid_instance_exist' in output_dict:
            total_loss = None
            loss_dict = {'loss':0.0,'contrast_shape_loss':0.0,'pts_match_loss':0.0,'match_gnode_loss':0.0}
        else:
            total_loss, loss_dict = loss_fn(loss_in_dict,data_dict,epoch)        
        output_dict['instance_matches'] = output_dict['node_matching']
        
        # 2. Ground-truth point matching
        if 'gt_points_matching_dict' in output_dict:
            gt_registration = register_scenes(
                                output_dict['gt_points_matching_dict'],
                                data_dict['batch_size'],
                                CORRS_ONLY=False)
            # if gt_registration['points'].shape[0]<1 and 'invalid_instance_exist' in output_dict:
            #     assert False
        else: 
            gt_registration = None
            
        # 3. Points match and registration
        if 'points_matching' in output_dict and 'invalid_instance_exist' not in output_dict:
            tictoc = TicToc()
            output_dict['registration'] = register_scenes(
                                output_dict['points_matching'],
                                data_dict['batch_size'])
            output_dict['time_list'].append(1000*tictoc.toc())
        with torch.no_grad():
            metric_dict, nodes_masks = summary_evaluation(data_dict, 
                                             output_dict, 
                                             gt_registration,
                                             conf.eval)
            output_dict['nodes_masks'] = nodes_masks

        return total_loss, loss_dict, output_dict, metric_dict

    def loss_fn(loss_in_dict,data_dict,epoch):        
        ## 1. gnn node match loss
        if cfg.loss.gnode_match_weight>0.0:
            loss_gnn_match, loss_gnn_dict = instance_matching_fn(
                loss_in_dict['node_matching'],data_dict)
            total_loss = loss_gnn_match
        else:
            loss_gnn_match = torch.tensor(0.0).to('cuda')
            total_loss = torch.tensor(0.0).to('cuda')
            
        ## 2. shape contrastive loss
        if cfg.loss.shape_contrast_weight>0.0 and \
            'shape_backbone' not in cfg.model.fix_modules:
            loss_contrast_shape = contrastive_loss_fn(data_dict,
                                    F.normalize(loss_in_dict['src_instance_shapes'],dim=1),
                                    F.normalize(loss_in_dict['ref_instance_shapes'],dim=1),
                                    cfg.loss.contrastive_postive_overlap,
                                    cfg.loss.contrastive_temp)
            total_loss += loss_contrast_shape
        else:
            loss_contrast_shape = torch.tensor(0.0).to('cuda')
        
        ## 3. fine point match loss
        if cfg.loss.fine_loss_weight>0.0 and \
            'gt_points_matching_dict' in loss_in_dict:
            loss_pts_match = point_matching_loss(
                loss_in_dict['gt_points_matching_dict'],
                data_dict['transform'])
            total_loss += loss_pts_match * cfg.loss.fine_loss_weight
        else:
            loss_pts_match = torch.tensor(0.0).to('cuda')
        
        with torch.no_grad():
            loss_dict = {
                'loss':total_loss.clone().detach().item(),
                'contrast_shape_loss':loss_contrast_shape.clone().detach().item(),
                'pts_match_loss':loss_pts_match.clone().detach().item(), 
                'match_gnode_loss':loss_gnn_match.clone().detach().item()}
            
        return total_loss, loss_dict

    def register_scenes(batch_matching_dict, 
                        B, 
                        CORRS_ONLY=False):
        '''
        Input,
        - batch_points_s: (M,K,3), source points
        - batch_points_t: (M,K,3), target points
        - batch_match_ptr: (M,), batch pointer
        - batch_pts_matching_scores: (M,K+1,K+1), matching scores
        Return,
        - batch: (C,), batch id
        - points: (C,6), [ref_x,ref_y,ref_z,src_x,src_y,src_z]
        '''

        batch_match_ptr = batch_matching_dict['matching_ptr']
        batch_matching_scores = batch_matching_dict['matching_scores']
        batch_points_ref = batch_matching_dict['ref_matching_points']
        batch_points_src = batch_matching_dict['src_matching_points']
        # Registration
        corres_points = []
        corres_scores = []
        corres_batch = []
        corres_instances = []
        estimated_transforms = []
        
        for scene_id in torch.arange(B):
            scene_mask = batch_match_ptr==scene_id 
            M_ = scene_mask.sum()
            if M_<1:
                estimated_transforms.append(torch.eye(4).unsqueeze(0).to('cuda'))
                continue    
            # fake_scores = torch.eye(conf.fine_matching.num_points_per_instance).to('cuda').unsqueeze(0).repeat(M_,1,1)
            assert torch.abs(batch_matching_scores[scene_mask].sum())>1e-6, 'matching scores should not be empty'
            corr_ref, corr_src, corr_scores, corr_indices0, estimated_transform = fine_registration(
                                            batch_points_ref[scene_mask],
                                            batch_points_src[scene_mask],
                                            batch_matching_scores[scene_mask],
                                            CORRS_ONLY)
            scene_matching_matrix = torch.exp(batch_matching_scores[scene_mask]) # (M,K+1,K+1)
            if scene_matching_matrix.shape[1]>batch_points_ref.shape[1]:
                scene_matching_matrix = scene_matching_matrix[:,:-1,:-1]
            ref_src_corres = torch.cat([corr_ref,corr_src],dim=1) # (C,6)
            
            corres_points.append(ref_src_corres)
            corres_instances.append(corr_indices0)
            corres_batch.append(scene_id.clone().repeat(ref_src_corres.shape[0]).to('cuda')) # (C_i,1)
            corres_scores.append(corr_scores) # (C_i,)
            estimated_transforms.append(estimated_transform.unsqueeze(0))
 
        corres_batch = torch.cat(corres_batch,dim=0) # (C,)
        corres_points = torch.cat(corres_points,dim=0) # (C,6), [ref_x,ref_y,ref_z,src_x,src_y,src_z]
        corres_instances = torch.cat(corres_instances,dim=0) # (C,)
        corres_scores = torch.cat(corres_scores,dim=0) # (C,)
        corres_errors = torch.zeros_like(corres_scores)
        estimated_transforms = torch.cat(estimated_transforms,dim=0) # (B,4,4)
        
        return {'batch':corres_batch.detach(), # (C,)
                'points':corres_points.detach(), # (C,6)
                'instances':corres_instances.detach(), # (C,)
                'scores':corres_scores.detach(), # (C,)
                'errors':corres_errors.detach(), # (C,)
                'corres_masks':torch.ones_like(corres_scores).detach(), # (C,)
                'estimated_transform':estimated_transforms.detach() # (B,4,4)
                }

    def calculate_shape_similarity(src_shape_feats, ref_shape_feats, data_dict):
        similarity_list = []
        
        for scan_id in torch.arange(data_dict['batch_size']):
            src_feats = src_shape_feats[data_dict['src_graph']['batch'][scan_id]:data_dict['src_graph']['batch'][scan_id+1]]
            ref_feats = ref_shape_feats[data_dict['ref_graph']['batch'][scan_id]:data_dict['ref_graph']['batch'][scan_id+1]]

            similarity = torch.einsum('mc,nc->mn',src_feats,ref_feats)
            similarity = similarity.detach().cpu()
            similarity_list.append(similarity)
        return similarity_list

    def summary_evaluation(data_dict:dict,
                           output_dict:dict,
                           gt_registration:dict,
                           eval_conf:dict):
        ''' Compute the metrics: 
        - nodes_tp, nodes_fp, nodes_gt
        - recall, precision, rmse, scenes
        - (optional) gt_precision
        '''
        metric_dict = {'nodes_tp':0,
                       'nodes_fp':0,
                       'nodes_gt':0,
                       'recall':0,
                       'rmse':0,
                       'scenes':0,
                       'precision':0
                        }
                            
        irs = torch.zeros(data_dict['batch_size']).float()
        if 'node_matching' not in output_dict: 
            return metric_dict          
        node_matching = output_dict['node_matching']

        if 'registration' in output_dict:
            registration_dict = output_dict['registration']     
        else: registration_dict = None 
        
        if gt_registration is not None: # eval pre-train
            metric_dict['gt_precision'] = 0.0
            for scan_id in torch.arange(data_dict['batch_size']):
                corres_points = gt_registration['points'][
                    gt_registration['batch']==scan_id]
                precision_, _ = evaluator.evaluate_fine(
                                        {'ref_corr_points':corres_points[:,:3],
                                         'src_corr_points':corres_points[:,3:],},
                                        {'transform':data_dict['transform'][scan_id]})
                metric_dict['gt_precision'] += precision_.detach().cpu().item()

        #
        pred_nodes_output = node_matching['pred_nodes'].detach()
        for scan_id in torch.arange(data_dict['batch_size']):
            # Node matches
            gt_iou_mat = data_dict['instance_ious'][scan_id]
            # if true_instances is None: continue
            scan_pred = pred_nodes_output[pred_nodes_output[:,0]==scan_id,1:]
            inst_tp, inst_fp, gt_pairs, inst_masks = eval_instance_match_new(gt_iou_mat,
                                                                   scan_pred,
                                                                   eval_conf.gt_node_iou)
            
            metric_dict['nodes_tp'] += inst_tp
            metric_dict['nodes_fp'] += inst_fp
            metric_dict['nodes_gt'] += gt_pairs # true_instances.shape[0]
            
            # Points matches
            if registration_dict is None: continue
            gt_transform = data_dict['transform'][scan_id]
            estimated_transform = registration_dict['estimated_transform'][scan_id,...]
            
            assert data_dict['transform'].ndim==3, 'transform should be 3D'
            assert registration_dict['estimated_transform'].ndim==3, 'est_transform should be 3D'
            corres_points = registration_dict['points'][
                registration_dict['batch']==scan_id]
            if corres_points.shape[0]<1:continue
            precision_, corres_rmse_ = evaluator.evaluate_fine(
                {'ref_corr_points':corres_points[:,:3],
                 'src_corr_points':corres_points[:,3:],},
                {'transform':gt_transform})
            corres_masks = torch.lt(corres_rmse_,evaluator.acceptance_radius).float()
            irs[scan_id] = precision_
            output_dict['registration']['errors'][
                registration_dict['batch']==scan_id] = corres_rmse_
            output_dict['registration']['corres_masks'][
                registration_dict['batch']==scan_id] = corres_masks
            
            # registration
            points = data_dict['batch_points'][scan_id]['points'][0].detach()
            ref_length = data_dict['batch_points'][scan_id]['lengths'][0][0].item()

            _, _, rmse, recall = evaluator.evaluate_registration(
                {'src_points':points[ref_length:],
                'estimated_transform':estimated_transform},
                {'transform':gt_transform})
            
            metric_dict['precision'] += precision_.detach().cpu().item()
            metric_dict['recall'] += recall.detach().cpu().item()
            metric_dict['rmse'] += rmse.detach().cpu().item()
            metric_dict['scenes'] += 1
        
        return metric_dict, inst_masks
        
    if test:
        return test_model_fn
    else:
        return model_fn