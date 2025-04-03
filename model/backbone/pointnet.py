import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
# from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from torch_geometric.typing import WITH_TORCH_CLUSTER
from omegaconf import OmegaConf

from dataset import SceneGraphDataList

if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet2(torch.nn.Module):
    '''PointNet++ classification network from the `"PointNet++: Deep
    Input: (N,4), [x,y,z,instance_id]
    '''
    def __init__(self, sample_ratio, shape_dim):
        super().__init__()
        if shape_dim<0:
            self.shape_dim=1024
        else:
            self.shape_dim = shape_dim
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(sample_ratio, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(sample_ratio, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, self.shape_dim], dropout=0.5, norm=None)

    # def forward(self, xyz, batch):   
    def forward(self, points_s, x_s_ptr, num_points_s):  

        xyz = points_s[:,:3]
        batch = self.create_batched_idx(points_s,x_s_ptr,num_points_s)
        max_batch_idx = batch.max().item()
        
        # print('{} points'.format(pos.shape[0])) 
        sa0_out = (None, xyz, batch)
        sa1_out = self.sa1_module(*sa0_out) # (N/r1,128),(N/r1,3),(N/r1,)
        sa2_out = self.sa2_module(*sa1_out) # (N/r2, 256), (N/r2, 3), (N/r2,)
        sa3_out = self.sa3_module(*sa2_out) # (B, 1024), (B, 3), (B,)
        feats, xyz, batch = sa3_out
        del sa3_out
        assert batch.shape[0]==max_batch_idx+1, 'batch size not aligned'

        return self.mlp(feats).log_softmax(dim=-1), batch
    
    def create_batched_idx(self,points_s,x_s_ptr,num_points_s):
        '''
        Input,
        - points_s: (N,4), [x,y,z,instance_id]
        - x_s_ptr: (B+1), instance id offset
        - num_points_s: (B,), [N1, N2, N3, ...], number of points in each scene scan
        Output,
        - pts_global_idx: (N,), batched instance index for each point.
        
        '''
        pts_local_idx = points_s[:,3].to(torch.long) # (N,)
        instances_id_offset = [x_s_ptr[scan_id].repeat(scan_num_pts) for scan_id, scan_num_pts in enumerate(num_points_s)] 
        pts_batched_idx = torch.cat(instances_id_offset,dim=0) + pts_local_idx # (N,)    
        return pts_batched_idx

class NCESoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, label):
        bsz = x.shape[0]
        # x = x.squeeze()
        loss = self.criterion(x, label)
        return loss

def ContrastiveDecorator(conf):
    
    def model_fn(batch_graph_pair,model,epoch):
        batch_graph_pair = batch_graph_pair.to('cuda')
        # global_idx_s = create_global_idx(batch_graph_pair.points_s,batch_graph_pair.x_s_ptr,batch_graph_pair.num_points_s)
        # global_idx_t = create_global_idx(batch_graph_pair.points_t,batch_graph_pair.x_t_ptr,batch_graph_pair.num_points_t)
        
        shapes_bs, _ = model(batch_graph_pair.points_s,batch_graph_pair.x_s_ptr,batch_graph_pair.num_points_s) # (m, dim_shape)
        shapes_bt, _ = model(batch_graph_pair.points_t,batch_graph_pair.x_t_ptr,batch_graph_pair.num_points_t) # (n, dim_shape)
        
        assert shapes_bs.shape[0]==batch_graph_pair.x_s.shape[0], 'batched shape embeddings not aligned'
        offset_matches = 0
        loss = None
        loss_out = {}
        
        # print('{} shapes_s, {} shapes_t, {} matches'.format(shapes_bs.shape[0],shapes_bt.shape[0],batch_graph_pair.matches.shape[0]))
        # return loss, loss_out
        
        for scan_id in torch.arange(batch_graph_pair.batch_size):
            nodes_mask_s = batch_graph_pair.x_s_batch==scan_id
            nodes_mask_t = batch_graph_pair.x_t_batch==scan_id
            
            # shape_s = shapes_bs[nodes_mask_s] # (m', dim_shape)
            # shape_t = shapes_bt[nodes_mask_t]  # (n', dim_shape)
            # q = shape_s[matches[:,0]] # (npos, dim_shape)
            # k = shape_t # (n', dim_shape)            
            if batch_graph_pair.num_matches[scan_id]<1:continue
            matches = batch_graph_pair.matches[offset_matches:offset_matches+batch_graph_pair.num_matches[scan_id]] # (npos,2)
            nce_loss = compute_loss(shapes_bs[nodes_mask_s][matches[:,0]], shapes_bt[nodes_mask_t], matches[:,1])
            
            if loss is None:
                loss = nce_loss
            else:
                loss = loss + nce_loss
            
            offset_matches = offset_matches + batch_graph_pair.num_matches[scan_id]

        with torch.no_grad():
            loss_out['loss'] = loss.clone().detach().item()

        return loss, loss_out
    
    def pointnet_mini_batch(points,global_idx,model,mb_size=60000):
        num_points = points.shape[0]
        num_insts = global_idx.max()+1
        mb_size = torch.tensor(mb_size)
        num_mb = torch.tensor(8).to(torch.long) #torch.floor(num_points/mb_size)+1
        mb_insts = torch.floor(num_insts/num_mb).to(torch.long)+1
        assert num_mb*mb_insts>=num_insts, 'check'
        
        shapes_b = torch.zeros((num_insts,model.shape_dim)).to('cuda')

        for i in torch.arange(num_mb):
            idx_s = i * mb_insts
            idx_e = min((i+1) * mb_insts, num_insts)
            mask = (global_idx>=idx_s) & (global_idx<idx_e)
            if mask.sum()<1:continue
            points_mb = points[mask]
            global_idx_mb = global_idx[mask]
            idx_s = global_idx_mb.min()
            idx_e = global_idx_mb.max()+1
            # valid_idx = torch.unique_consecutive(global_idx_mb)
            if points_mb.shape[0]>mb_size:
                print('raw batch size {}, large mb size {} in threshod {}'.format(points.shape[0],points_mb.shape[0],mb_size))
            
            # shapes_mb = model(points_mb[:,:3],global_idx_mb)[idx_s:]
            shapes_b[idx_s:idx_e] = model(points_mb[:,:3],global_idx_mb)[idx_s:]
        
        return shapes_b
    
    # def create_global_idx(x_s_ptr,num_points_s,points_s):
    #     # pts_pos = points_s[:,:3] # (N,3)
    #     pts_local_idx = points_s[:,3].to(torch.long) # (N,)
    #     instances_id_offset = [x_s_ptr[scan_id].repeat(scan_num_pts) for scan_id, scan_num_pts in enumerate(num_points_s)] 
    #     pts_global_idx = torch.cat(instances_id_offset,dim=0) + pts_local_idx # (N,)    
    #     return pts_global_idx

    def compute_loss(q, k, labels, mask=None):
        ''' Adopted from Hou Ji.
            q: (npos, dim_shape)
            k: (nkeys, dim_shape)
            labels: (npos,) in range [0,nkeys)
        '''
        nceT = conf.nceT
        # npos = q.shape[0] 
        logits = torch.mm(q, k.transpose(1, 0)) # npos by npos
        # labels = torch.arange(npos).cuda().long()
        out = torch.div(logits, nceT)
        out = out.contiguous()
        if mask != None:
            raise NotImplementedError
            out = out - LARGE_NUM * mask.float()
        criterion = NCESoftmaxLoss().cuda()
        loss = criterion(out, labels)
        return loss


    return model_fn

if __name__=='__main__':
    dataroot = '/data2/ScanNetGraph'
    conf = OmegaConf.load('config/scannet.yaml')
    train_data = SceneGraphDataList(dataroot=dataroot,split='val',conf=conf)
    train_loader = DataLoader(train_data.graph_data_list, batch_size=conf.train.batch_size,follow_batch=['x_s', 'x_t'])
    print('find {} pairs of scene graph and {} batch'.format(train_data.len(),len(train_loader)))
    
    # x_s_batch: (num_s_nodes,), scan_id_in_batch
    batch = next(iter(train_loader))
    print(batch)
    
    # Model 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNet2().to(device)
    
    #
    print('running pointnet++')
    batch = batch.to(device)
    pts_pos = batch.points_s[:,:3] # (N,3)
    pts_local_idx = batch.points_s[:,3].to(torch.long) # (N,)
    pts_offset = [batch.x_s_ptr[scan_id].repeat(scan_num_pts) for scan_id, scan_num_pts in enumerate(batch.num_points_s)] 
    pts_global_idx = torch.cat(pts_offset,dim=0) + pts_local_idx # (N,)
    print('range [{}-{}]'.format(pts_global_idx.min(),pts_global_idx.max()))
    out = model(pts_pos,pts_global_idx) # (batch_insts, dim_shape)
    print(out.shape)
    assert out.shape[0]==batch.x_s.shape[0]
    
    