import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointGNNConv,MLP,GATv2Conv, SAGEConv
# from model.gnn.attentions import CrossBlock
from model.gnn.triplet_gnn import TripletGNN, GetCornerIndices
from model.gnn.spatial_attention import SpatialTransformer, CrossBlock
import time
from model.utils.tictoc import TicToc

def create_all_edges(feats, batches):
    '''
    Create all edges for the graph topology.
    Input:
        - feats: (N1+N2+...+NB, in_channels)
        - batches: (B+1)
    '''
    batch_size = batches.shape[0]-1
    n_s = feats.shape[0]
    
    affinity_matrix = torch.zeros((n_s,n_s)).long().to(feats.device)
    for scan_id in torch.arange(batch_size):
        k0, k1 = batches[scan_id], batches[scan_id+1]
        affinity_matrix[k0:k1,k0:k1] = 1
    affinity_matrix = affinity_matrix.masked_fill(torch.eye(n_s).bool().to(feats.device),0) # remove self-edges
    all_edges = torch.nonzero(affinity_matrix).t() # (2,e)

    return all_edges.long()

class PointGCN(nn.Module):
    def __init__(self, input_dim, edge_dim, out_dim):
        super(PointGCN, self).__init__()
        self.gnn_layer = PointGNNConv(mlp_h=MLP([input_dim,32,3]),
                                        mlp_f=MLP([input_dim+3,edge_dim,edge_dim]),
                                        mlp_g=MLP([edge_dim,out_dim,out_dim]))
    def forward(self, x, pos, edges):
        edges_ = edges.clone().long()
        x = self.gnn_layer(x, pos, edges_.t())
        return x

class SelfGAT(nn.Module):
    def __init__(self, in_channels, out_channels, aggregate_all_edges, heads=4, dropout=0.2):
        super(SelfGAT, self).__init__()
        self.aggregate_all_edges = aggregate_all_edges
        self.cross_gat_layer1 = GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout)
        # self.cross_gat_layer2 = GATv2Conv(in_channels*heads, out_channels, heads=1, dropout=dropout)    

    def forward(self, feats, edges, batches):
        '''
        Input,
            - feats: (N,D)
            - edges: (E,2)
        '''
        n_s = feats.shape[0]
        batch_size = batches.shape[0]-1
        
        # if self.aggregate_all_edges:
        #     affinity_matrix = torch.zeros((n_s,n_s)).long().to(feats.device)
        #     for scan_id in torch.arange(batch_size):
        #         k0, k1 = batches[scan_id], batches[scan_id+1]
        #         affinity_matrix[k0:k1,k0:k1] = 1
        #     affinity_matrix = affinity_matrix.masked_fill(torch.eye(n_s).bool().to(feats.device),0) # remove self-connections
        #     all_edges = torch.nonzero(affinity_matrix).t() # (2,e)
        #     gnn_edges = all_edges
        # else:
        gnn_edges = edges.clone().long() # (2,e)
        feats, atten_weights = self.cross_gat_layer1(feats,gnn_edges,return_attention_weights=True)
        # feats = F.elu(feats)
        # feats = self.cross_gat_layer2(feats,gnn_edges)
        
        return feats
        


class CrossGraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2):
        super(CrossGraphLayer, self).__init__()
        self.cross_gat_layer1 = GATv2Conv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.cross_gat_layer2 = GATv2Conv(in_channels*heads, out_channels, heads=1, dropout=dropout)
    
    def forward(self, x_s, x_t, src_batch, ref_batch):
        '''
        Input,
            - src_batch: (B+1)
            - tar_batch: (B+1)
        '''
        n_s = x_s.shape[0]
        n_t = x_t.shape[0]
        # D = x_s.shape[1]
        batch_size = src_batch.shape[0]-1
        
        affinity_matrix = torch.zeros((x_s.shape[0],x_t.shape[0])).to('cuda') # all valid nodes between source and target are connected
        # affinity_matrix[gt[:,0],gt[:,1]] = 1 # for debug, set the true matches to affinity.
        for scan_id in torch.arange(batch_size):
            k0_s = src_batch[scan_id]
            k1_s = src_batch[scan_id+1]
            k0_t = ref_batch[scan_id]
            k1_t = ref_batch[scan_id+1]
            affinity_matrix[k0_s:k1_s,k0_t:k1_t] = 1
        cross_edges_unidir = affinity_matrix.nonzero().t() # (2,e), [i_s,i_t]
        cross_edges_unidir[1,:] = cross_edges_unidir[1,:] + x_s.shape[0]
        cross_edges_bidir = torch.cat([cross_edges_unidir,cross_edges_unidir[[1,0],:]],dim=1) # (2,2e)

        x_s_t = torch.cat([x_s,x_t],dim=0) # (m+n,feat_dim)
        x_s_t, attention_weights = self.cross_gat_layer1(x_s_t,cross_edges_bidir,return_attention_weights=True) # (m+n,feat_dim * heads)
        atten_edge_index, atten_edge_weights = attention_weights # (2,2e+m+n), (2e+m+n,H)
        x_s_t = F.elu(x_s_t)
        x_s_t = self.cross_gat_layer2(x_s_t,cross_edges_bidir) # (m+n,feat_dim)

        assert x_s_t.shape[1] == x_s.shape[1], 'feature dimension not match'        
        assert x_s_t.shape[0] == x_s.shape[0] + x_t.shape[0], 'node number not match'
        x_s = x_s_t[:n_s,:]
        x_t = x_s_t[n_s:,:]
        
        return x_s, x_t

class GraphNeuralNetwork(nn.Module):
    def __init__(self, conf):
        super(GraphNeuralNetwork, self).__init__()
        
        self.gnn_layers = nn.ModuleDict()
        self.gnn_node_dim = conf.semantic_dim
        self.aggregate_all_edges = conf.gnn.all_self_edges
        if 'triplet_number' in conf.gnn:
            self.triplet_number = conf.gnn.triplet_number
        else:
            self.triplet_number = 20
        for layer in conf.gnn.layers:
            if layer=='self':
                self.gnn_layers[layer] = SelfGAT(conf.node_dim,conf.node_dim, conf.gnn.all_self_edges, conf.gnn.heads)
            elif layer=='gtop' or layer=='ltriplet':
                self.gnn_layers[layer] = TripletGNN(conf.node_dim,conf.gnn.hidden_dim,conf.node_dim,conf.gnn.reduce,
                                                          conf.gnn.enable_dist_embedding,conf.gnn.triplet_mlp,conf.gnn.triplet_activation)
            elif layer=='gtriplet':
                self.gnn_layers[layer] = TripletGNN(conf.node_dim,conf.gnn.hidden_dim,conf.node_dim,conf.gnn.reduce,
                                                          conf.gnn.enable_dist_embedding)
            elif layer=='sage':
                self.gnn_layers[layer] = SAGEConv(self.gnn_node_dim,self.gnn_node_dim)
            elif layer=='sattn' or layer=='gsattn':
                self.gnn_layers[layer] = SpatialTransformer(conf.node_dim,conf.gnn.heads,conf.gnn.position_encoding,conf.gnn.all_self_edges)
            elif layer=='pointgcn':
                self.gnn_layers[layer] = PointGCN(conf.node_dim,conf.node_dim, conf.node_dim)
            elif layer=='crossblock':
                self.gnn_layers[layer] = CrossBlock(conf.node_dim,conf.gnn.heads,flash=True)
            else:
                raise NotImplementedError

    def reorganize_nodes(self,scan_x_src,K):
        '''
        Input, 
            - scan_x_src: (N,D)
            - K: int
        '''
        n_src = scan_x_src.shape[0]
        if n_src<=K:
            scan_x_src = torch.cat([scan_x_src,torch.zeros((K-n_src,scan_x_src.shape[1])).to(scan_x_src.device)],dim=0) # (K,D)
            scan_mask_src = torch.cat([torch.ones((n_src)),torch.zeros((K-n_src))],dim=0).to(scan_x_src.device) # (K)
            indices_src = torch.arange(n_src).to(scan_x_src.device)
        else:
            indices_src = torch.randperm(n_src)[:K].to(scan_x_src.device) # (K)
            scan_x_src = scan_x_src[:K]
            scan_mask_src = torch.ones(K).to(scan_x_src.device)
        return scan_x_src, scan_mask_src,indices_src
    
    
    def forward(self, x_src, x_ref, batch_graph_dict):
        # torch.autograd.set_detect_anomaly(True)
        time_record = []
        tictoc = TicToc()
        
        for layer_name, layer in self.gnn_layers.items():
            src_edges_bd = batch_graph_dict['src_graph']['edge_indices'].t().long() # (2,e), bi-directional
            ref_edges_bd = batch_graph_dict['ref_graph']['edge_indices'].t().long() # (2,e)
            src_global_edges = batch_graph_dict['src_graph']['global_edge_indices'].t() # (2,e)
            ref_global_edges = batch_graph_dict['ref_graph']['global_edge_indices'].t() # (2,e)
            if self.aggregate_all_edges:
                src_edges_bd = create_all_edges(x_src,batch_graph_dict['src_graph']['batch'])
                ref_edges_bd = create_all_edges(x_ref,batch_graph_dict['ref_graph']['batch'])
            
            if layer_name=='self':
                x_src = layer(x_src,src_edges_bd,batch_graph_dict['src_graph']['batch'])
                x_ref = layer(x_ref,ref_edges_bd,batch_graph_dict['ref_graph']['batch'])
            elif layer_name=='ltriplet':
                time_record.append(tictoc.toc())
                src_anchors,src_triplets,src_corner_masks = GetCornerIndices(batch_graph_dict['src_graph']['centroids'],
                                                                             src_edges_bd,'zero',
                                                                             self.triplet_number)                  
                ref_anchors,ref_triplets,ref_corner_masks = GetCornerIndices(batch_graph_dict['ref_graph']['centroids'],
                                                                             ref_edges_bd,'zero',
                                                                             self.triplet_number)                
                time_record.append(tictoc.toc())
                if src_anchors.shape[0]>0:
                    x_src,src_checkmask = layer(x_src, batch_graph_dict['src_graph']['centroids'],
                                  src_anchors,src_triplets,src_corner_masks)
                    assert src_checkmask.sum()<3, '{} triplet not orderred properly'.format(src_checkmask.sum())                    
                else:   
                    x_src = torch.cat([x_src,torch.zeros_like(x_src)],dim=1)

                if ref_anchors.shape[0]>0:
                    x_ref,ref_checkmask = layer(x_ref, batch_graph_dict['ref_graph']['centroids'],
                                  ref_anchors,ref_triplets,ref_corner_masks)          
                    assert ref_checkmask.sum()<3, '{} triplet not orderred properly'.format(ref_checkmask.sum())          
                else:
                    x_ref = torch.cat([x_ref,torch.zeros_like(x_ref)],dim=1)
                time_record.append(tictoc.toc())
                
            elif layer_name=='gtriplet':
                src_out = layer(x_src, batch_graph_dict['src_graph']['centroids'], src_global_edges)
                ref_out = layer(x_ref, batch_graph_dict['ref_graph']['centroids'], ref_global_edges)
            elif layer_name=='sage':
                x_src = layer(x_src,src_edges_bd)
                x_ref = layer(x_ref,ref_edges_bd)
            elif layer_name=='sattn':
                x_src = layer(x_src,batch_graph_dict['src_graph']['centroids'],src_edges_bd,batch_graph_dict['src_graph']['batch'])
                x_ref = layer(x_ref,batch_graph_dict['ref_graph']['centroids'],ref_edges_bd,batch_graph_dict['ref_graph']['batch'])
            elif layer_name=='gsattn':
                x_src = layer(x_src,batch_graph_dict['src_graph']['centroids'],src_global_edges,batch_graph_dict['src_graph']['batch'])
                x_ref = layer(x_ref,batch_graph_dict['ref_graph']['centroids'],ref_global_edges,batch_graph_dict['ref_graph']['batch'])
            elif layer_name=='pointgcn':
                x_src = layer(x_src,batch_graph_dict['src_graph']['centroids'],src_edges_bd)
                x_ref = layer(x_ref,batch_graph_dict['ref_graph']['centroids'],ref_edges_bd)
            elif layer_name=='crossblock':
                x_src_batch = []
                x_ref_batch = []
                src_indices = []
                ref_indices = []
                mask_batch = []
                K = 60
                
                # todo
                for scan_id in torch.arange(batch_graph_dict['batch_size']):
                    src_masks = batch_graph_dict['src_graph']['scene_mask']==scan_id
                    ref_masks = batch_graph_dict['ref_graph']['scene_mask']==scan_id
                    
                    scan_x_src, scan_mask_src, _ = self.reorganize_nodes(x_src[src_masks],K) # (K,D), (K), (-1)
                    scan_x_ref, scan_mask_ref, _ = self.reorganize_nodes(x_ref[ref_masks],K)

                    x_src_batch.append(scan_x_src.unsqueeze(0))
                    x_ref_batch.append(scan_x_ref.unsqueeze(0))
                    mask_batch.append((scan_mask_src.unsqueeze(1) * scan_mask_ref.unsqueeze(0).unsqueeze(0))) # (1,K,K)
                    # src_indices.append(indices_src)
                    # ref_indices.append(indices_ref)
                
                x_src_batch = torch.cat(x_src_batch,dim=0) # (B,K,D)
                x_ref_batch = torch.cat(x_ref_batch,dim=0) # (B,K,D)
                mask_batch = torch.cat(mask_batch,dim=0).bool() # (B,K,K)
                mask_batch = mask_batch.unsqueeze(1) # (B,1,K,K)
                
                assert x_src_batch.ndim==3 and x_ref_batch.ndim==3 and mask_batch.ndim==4, 'x_src_batch and x_ref_batch should be 3D'
                x_src_batch, x_ref_batch = layer(x_src_batch,x_ref_batch,mask_batch) # (B,K,D), (B,K,D)
                
                for scan_id in torch.arange(batch_graph_dict['batch_size']):
                    src_masks = batch_graph_dict['src_graph']['scene_mask']==scan_id
                    ref_masks = batch_graph_dict['ref_graph']['scene_mask']==scan_id
                    if src_masks.sum()>K:
                        x_src[:K] = x_src_batch[scan_id]
                    else:
                        x_src[src_masks] = x_src_batch[scan_id][:src_masks.sum()]
                        
                    if ref_masks.sum()>K:
                        x_ref[:K] = x_ref_batch[scan_id]
                    else:
                        x_ref[ref_masks] = x_ref_batch[scan_id][:ref_masks.sum()]

            # assert torch.all(torch.isnan(x_src)==False) and torch.all(torch.isnan(x_ref)==False), 'Nan features after encoded {}'.format(layer_name)
        
        src_checknan = torch.isnan(x_src).sum(dim=1)
        ref_checknan = torch.isnan(x_ref).sum(dim=1)
        if src_checknan.sum()>0 or ref_checknan.sum()>0:
            print('Nan features after encoded')
            print('{} nan src and {} nan ref!'.format(src_checknan.sum(),ref_checknan.sum()))
            # assert False
        
        # 
        # msg = ['{:.2f} ms'.format(1000*t) for t in time_record]
        # print('GNN time: ', ' '.join(msg))
        
        return x_src,x_ref,time_record[2]