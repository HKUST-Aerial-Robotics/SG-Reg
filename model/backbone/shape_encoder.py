import sys
import torch
import torch.nn as nn
from model.ops import sample_instance_from_points
from model.kpconv import ResidualBlock, LastUnaryBlock
from model.utils.tictoc import TicToc

class KPConvShape(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 kernel_size, 
                 init_radius, 
                 init_sigma, 
                 group_norm,
                 K_shape=1024,
                 K_match=256,
                 decoder=True):
        super(KPConvShape, self).__init__()

        self.encoder1_1 = ResidualBlock(
            input_dim, output_dim, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        
        if decoder:
            self.decoder = LastUnaryBlock(output_dim+input_dim, input_dim)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.K_shape = K_shape
        self.K_match = K_match
            
    def forward(self, f_points, 
                    f_feats, 
                    f_instances, 
                    instances_centroids, 
                    instances_points_indices,
                    decode_points):             
        '''
        Input:
            f_points: (P, 3), 
            f_feats: (P, C), 
            f_instances: (P, 1), 
            instances_centroids: (N, 3), instance centroid poistion
            instances_points_indices: (N, H), instance knn indices
        '''   
        
        #! encoder reduce feature dimension C->C/4
        instance_feats = self.encoder1_1(f_feats, instances_centroids, f_points, instances_points_indices) # (N, C)
        assert instance_feats.shape[0] == instances_centroids.shape[0]
        if decode_points: # This is only used in pre-train constrastive learning. In validation and deployment, skip it.
            f_instance_feats = instance_feats[f_instances.squeeze()] # (P, C)
            feats_decoded = self.decoder(torch.cat([f_instance_feats, f_feats], dim=1))
            assert feats_decoded.shape[0] == f_points.shape[0]
            return instance_feats, feats_decoded
        else:
            return instance_feats, None

def concat_instance_points(ref_instance_number:torch.Tensor,
                           src_instance_number:torch.Tensor,
                           ref_instance_points:torch.Tensor,
                           src_instance_points:torch.Tensor,
                           device:torch.device):
    ''' build instance-labeled points and instance list.
        Return:
            instance_list: (N+M,), instance_points: (P+Q, 3)
    '''
    ref_instance_list = torch.arange(ref_instance_number).to(device)
    src_instance_list = torch.arange(src_instance_number).to(device)+ref_instance_number
    src_instance_points = src_instance_points + ref_instance_number
    instance_list = torch.cat([ref_instance_list,src_instance_list],dim=0) # (N+M,)
    instance_points = torch.cat([ref_instance_points,src_instance_points],dim=0) # (P+Q, 3)
    
    return instance_list, instance_points

def encode_batch_scenes_instances(shape_backbone:nn.Module,
                                    batch_graph_pair:dict, 
                                    instance_f_feats_dict:dict, 
                                    decode_points=True, 
                                    verify=True):
        tictoc = TicToc()
        stack_instances_shape = []
        stack_instance_pts_match = []
        stack_instances_batch = [torch.tensor(0).long()]
        stack_feats_f_decoded = []
        ref_graph_batch = batch_graph_pair['ref_graph']['batch']
        src_graph_batch = batch_graph_pair['src_graph']['batch']
        invalid_instance_exist = False
        duration_list = []
        
        for scene_id in torch.arange(batch_graph_pair['batch_size']):
            # Extract scene data
            data_dict = batch_graph_pair['batch_points'][scene_id]
            f_pts_length = data_dict['lengths'][1] # [P,Q]
            assert f_pts_length.device == data_dict['insts'][1].device, 'f_pts_length and insts device are not match'
            assert data_dict['insts'][1].is_contiguous(), 'insts[1] is not contiguous'
            tmp_instances_f = data_dict['insts'][1]
            # todo: this step is slow. approx. 50ms
            ref_instances_f = tmp_instances_f[:f_pts_length[0]] # (P,)
            src_instances_f = tmp_instances_f[f_pts_length[0]:] # (Q,) 
            points_f = data_dict['points'][1] # (P+Q, 3)
            duration_list.append(tictoc.toc())

            feats_b0 = instance_f_feats_dict['feats_batch'][scene_id]
            feats_b1 = instance_f_feats_dict['feats_batch'][scene_id+1]
            feats_f = instance_f_feats_dict['feats_f'][feats_b0:feats_b1] # (P+Q, C)
            
            instance_list, instances_f = concat_instance_points(
                                            ref_graph_batch[scene_id+1]-ref_graph_batch[scene_id],
                                            src_graph_batch[scene_id+1]-src_graph_batch[scene_id],
                                            ref_instances_f, 
                                            src_instances_f, 
                                            feats_f.device)
            assert points_f.shape[0] == instances_f.shape[0] \
                and points_f.shape[0]==feats_f.shape[0], 'points, feats, and instances shape are not match'
            
            # 
            instance_fpts_indxs_shape, invalid_instance_mask = \
                sample_instance_from_points(instances_f, instance_list, 
                                            shape_backbone.K_shape, points_f.shape[0]) # (N+M,Ks), (N+M,)
            instance_fpts_indxs_match, _ =\
                sample_instance_from_points(instances_f, instance_list, 
                                            shape_backbone.K_match, points_f.shape[0]) # (N+M,Km), (N+M,)
            if(torch.any(invalid_instance_mask)): # involves invalid instances
                print('instance without points in {}'.format(batch_graph_pair['src_scan'][scene_id]))
                # assert False, 'Some instances are not assigned to any fine points.'
                instances_shape = torch.zeros((instance_list.shape[0],
                                               shape_backbone.output_dim)).to(feats_f.device)
                feats_f_decoded = torch.zeros((feats_f.shape[0],
                                               feats_f.shape[1])).to(feats_f.device)
                invalid_instance_exist = True
            else: # shape instance-wise shapes and points
                # Extract instance centroids
                instance_f_points = points_f[instance_fpts_indxs_shape] # (N+M, Ks, 3)
                instance_f_centers = instance_f_points.mean(dim=1) # (N+M, 3)    

                if verify:
                    ref_instance_centroids = batch_graph_pair['ref_graph']['centroids'][batch_graph_pair['ref_graph']['scene_mask']==scene_id]
                    src_instance_centroids = batch_graph_pair['src_graph']['centroids'][batch_graph_pair['src_graph']['scene_mask']==scene_id]
                    instance_centroids = torch.cat([ref_instance_centroids,src_instance_centroids],dim=0) # (M+N, 3)
                    dist = torch.abs((instance_centroids - instance_f_centers).mean(dim=1)) # (M+N,)
                    assert dist.max()<1.0, 'Center distance is too large {:.4f}m'.format(dist.max())   

                instances_shape, feats_f_decoded = shape_backbone(points_f, 
                                                                feats_f, 
                                                                instances_f, 
                                                                instance_f_centers, 
                                                                instance_fpts_indxs_shape,
                                                                decode_points)
            duration_list.append(tictoc.toc())

            stack_instances_shape.append(instances_shape)
            stack_instance_pts_match.append(instance_fpts_indxs_match)
            stack_instances_batch.append(stack_instances_batch[-1]+instance_list.shape[0])

            if decode_points:
                assert feats_f_decoded is not None, 'feats_f_decoded is None'
                stack_feats_f_decoded.append(feats_f_decoded) # (P+Q, C0)        
        
        # Concatenate all scenes
        stack_instances_shape = torch.cat(stack_instances_shape,dim=0) # (M+N, C1)
        stack_instances_shape = nn.functional.normalize(stack_instances_shape, dim=1) # (M+N, C1)
        instance_f_feats_dict['instances_shape'] = stack_instances_shape
        instance_f_feats_dict['instances_f_indices_match'] = torch.cat(stack_instance_pts_match,dim=0)
        instance_f_feats_dict['instances_batch'] = torch.stack(stack_instances_batch,dim=0) # (B+1,)
        instance_f_feats_dict['invalid_instance_exist'] = invalid_instance_exist

        if decode_points:
            stack_feats_f_decoded = torch.cat(stack_feats_f_decoded,dim=0) # (P+Q, C0)
            instance_f_feats_dict['feats_f_decoded'] = stack_feats_f_decoded
        duration_list.append(tictoc.toc())
        
        # timing
        # msg = ['{:.2f}'.format(1000*t) for t in duration_list]
        # print('Shape Encoder: ', ' '.join(msg))     
             
        return instance_f_feats_dict, duration_list[1]