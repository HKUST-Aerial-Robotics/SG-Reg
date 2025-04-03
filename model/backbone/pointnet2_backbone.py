import sys, os
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from pointnet2.pointnet2_batch import pointnet2_modules
from pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()
        xyz = xyz.view(batch_size, -1, 3)
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1).contiguous() if features is not None else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        point_features = l_features[0].permute(0, 2, 1).contiguous()  # (B, N, C)
        batch_dict['point_features'] = point_features.view(-1, point_features.shape[-1])
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0].view(-1, 3)), dim=1)
        return batch_dict


class PointNet2Backbone(nn.Module):
    """
    DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723
    """
    def __init__(self, model_cfg, input_channels, **kwargs):
        # assert False, 'DO NOT USE THIS CURRENTLY SINCE IT MAY HAVE POTENTIAL BUGS, 20200723'
        super().__init__()
        self.model_cfg = model_cfg

        # SA layers
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.num_points_each_layer.append(self.model_cfg.SA_CONFIG.NPOINTS[k])
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy() # array (2,3)
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules_stack.StackSAModuleMSG(
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        # FP layers
        self.FP_modules = nn.ModuleList()
        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules_stack.StackPointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )
        # Global aggregation layer
        assert self.model_cfg.GLOBAL_SA.MLPS[0][0] == self.model_cfg.SA_CONFIG.MLPS[-1][0][-1]+self.model_cfg.SA_CONFIG.MLPS[-1][1][-1]
        self.global_sa = pointnet2_modules_stack.StackSAModuleMSG(
            radii=self.model_cfg.GLOBAL_SA.RADIUS,
            nsamples=self.model_cfg.GLOBAL_SA.NSAMPLE,
            mlps=self.model_cfg.GLOBAL_SA.MLPS.copy(),
            use_xyz=True
        )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        l_xyz, l_features, l_batch_cnt = [xyz], [features], [xyz_batch_cnt]
        for i in range(len(self.SA_modules)):
            new_xyz_list = []
            for k in range(batch_size):
                if len(l_xyz) == 1:
                    cur_xyz = l_xyz[0][batch_idx == k]
                else:
                    last_num_points = self.num_points_each_layer[i - 1]
                    cur_xyz = l_xyz[-1][k * last_num_points: (k + 1) * last_num_points]
                cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(
                    cur_xyz[None, :, :].contiguous(), self.num_points_each_layer[i]
                ).long()[0]
                if cur_xyz.shape[0] < self.num_points_each_layer[i]:
                    # empty_num = self.num_points_each_layer[i] - cur_xyz.shape[0]
                    # cur_pt_idxs[-empty_num:] = cur_pt_idxs[:empty_num].clone()
                    times = int(self.num_points_each_layer[i] / cur_xyz.shape[0]) + 1
                    non_empty = cur_pt_idxs[:cur_xyz.shape[0]].clone()      
                    cur_pt_idxs = non_empty.repeat(times)[:self.num_points_each_layer[i]]            
                new_xyz_list.append(cur_xyz[cur_pt_idxs])
            new_xyz = torch.cat(new_xyz_list, dim=0)

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(self.num_points_each_layer[i])
            li_xyz, li_features = self.SA_modules[i](
                xyz=l_xyz[i], features=l_features[i], xyz_batch_cnt=l_batch_cnt[i],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt
            )

            l_xyz.append(li_xyz)
            l_features.append(li_features)
            l_batch_cnt.append(new_xyz_batch_cnt)
        
        l_features[0] = points[:, 1:]
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                unknown=l_xyz[i - 1], unknown_batch_cnt=l_batch_cnt[i - 1],
                known=l_xyz[i], known_batch_cnt=l_batch_cnt[i],
                unknown_feats=l_features[i - 1], known_feats=l_features[i]
            )

        # Gloabl feature. Encode a feature feature for each scan in the batch.
        coarse_batch_cnt = l_batch_cnt[-1] # (B,)
        assert coarse_batch_cnt.min() == coarse_batch_cnt.max()
        global_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(1)
        global_xyz = []
        for k in range(batch_size): # Sample one point from each scan
            cur_xyz = l_xyz[-1][k*coarse_batch_cnt[0]:(k+1)*coarse_batch_cnt[0]]
            cur_pt_idxs = pointnet2_utils_stack.farthest_point_sample(
                cur_xyz[None, :, :].contiguous(), 1
            ).long()[0]
            global_xyz.append(cur_xyz[cur_pt_idxs])
        global_xyz = torch.cat(global_xyz, dim=0) # (B, 3)
        global_xyz, global_feature = self.global_sa(
            xyz=l_xyz[-1], features=l_features[-1], xyz_batch_cnt=coarse_batch_cnt,
            new_xyz=global_xyz, new_xyz_batch_cnt=global_xyz_batch_cnt
        )
        assert global_feature.shape[0] == batch_size

        batch_dict['point_features'] = l_features[0] # (num_points, num_point_features)
        batch_dict['point_coords'] = torch.cat((batch_idx[:, None].float(), l_xyz[0]), dim=1) # (num_points, 4)
        # batch_dict['coarse_features'] = l_features[-1]
        batch_dict['global_features'] = global_feature
        return batch_dict


if __name__ == "__main__":
    from torch.autograd import Variable
    from omegaconf import OmegaConf
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)
    
    cfg_file = 'config/scannet.yaml'
    cfg = OmegaConf.load(cfg_file)
    
    print(cfg)
    net = PointNet2MSG(cfg.backbone.pointnet, input_channels=3)
    # net = PointNet2Backbone(cfg.backbone.pointnet, input_channels=3)
    net.cuda()
    net.eval()
    print('Built model')
    
    # Generate data 
    object_size = 8000
    nProposals = 200
    N = object_size * nProposals
    # xyz = torch.randn(N,3).cuda()
    # batches = torch.randint(0, nProposals, (N,)).cuda() 
    # batches = batches.sort()[0]
    # points = torch.cat([batches[:, None].float(),xyz], dim=1)
    
    points = []
    for i in torch.arange(nProposals):
        xyz = torch.randn(object_size,3).cuda()
        xyz = torch.cat([i*torch.ones(object_size,1).cuda(),xyz], dim=1)
        points.append(xyz)
    points = torch.cat(points, dim=0)
    
    out = net({'batch_size':nProposals,'points':points})
    
    points_feats = out['point_features'] # (N, C0)
    points_coords = out['point_coords'] # (N, 4)
    # global_feats = out['global_features'] # (nProposals, C_global)
    batch_out = points_coords[:,0] # (N,),[0,1,...,nProposals-1]
    print('points_feats', points_feats.shape)
    print('points_coords', points_coords.shape)
    # print('global features', global_feats.shape)
    print('batch range:', batch_out.min(), batch_out.max())
    
    diff = points - points_coords
    print('diff: ', diff.sum())
    