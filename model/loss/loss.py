import sys
import torch
import torch.nn as nn
from model.ops import apply_transform, pairwise_distance
sys.path.append('/home/cliuci/code_ws/GeoTransformer')

from geotransformer.modules.loss import WeightedCircleLoss

def contrastive_loss_fn(data_dict:dict,
                        src_stack_feats:torch.Tensor,
                        ref_stack_feats:torch.Tensor,
                        positive_overlap=0.1,
                        temp=0.1):
# (data_dict:dict,scores:list, positive_overlap=0.1, temp=0.1):
    
    def run_jl(embed, embed_neg, pseudo_labels, pseudo_labels_neg, pos_idx1, pos_idx2, neg_idx,temp=0.1):
        '''
        From JiangLi's code
        https://github.com/llijiang/GuidedContrast/blob/main/model/semseg/semseg.py#L209
        '''
        neg = (embed @ embed_neg.T) / temp
        mask = torch.ones((embed.shape[0], embed_neg.shape[0]), dtype=torch.float32, device=embed.device)
        mask *= ((neg_idx.unsqueeze(0) != pos_idx1.unsqueeze(-1)).float() *
                    (neg_idx.unsqueeze(0) != pos_idx2.unsqueeze(-1)).float())
        pseudo_label_guidance = (pseudo_labels.unsqueeze(-1) != pseudo_labels_neg.unsqueeze(0)).float()
        mask *= pseudo_label_guidance
        neg = (torch.exp(neg) * mask).sum(-1)
        return neg
        
    def run(pos_feats1, pos_feats2, neg_feats, neg_mask, pos1_labels, neg_labels, temp):
        '''
        Input, 
            - pos_feats1: (N, D), torch.Tensor
            - pos_feats2: (N, D), torch.Tensor
            - neg_feats: (M, D), torch.Tensor
            - neg_mask: (N,M), torch.Tensor
            - size_diff, (N,M), torch.Tensor #todo
            - pos_labels: (N,), list
            - neg_labels: (M,), list
        '''
        pos = (pos_feats1 * pos_feats2).sum(dim=-1,keepdim=True) / temp # (N,1)
        pos = torch.exp(pos).squeeze()
        
        #        
        psudo_labels = torch.zeros_like(neg_mask, dtype=torch.int32) # (N,M), different semantics are set to 1
        for row, src_label in enumerate(pos1_labels):
            for col, ref_label in enumerate(neg_labels):
                if src_label!=ref_label:
                    psudo_labels[row,col] = 1
        neg_mask = neg_mask * psudo_labels
        
        #
        # neg = (feats_dist*neg_mask).sum(dim=1)  # (N,)
        neg = (pos_feats1 @ neg_feats.T) / temp
        neg = torch.exp(neg) # (N,M)
        neg = (neg * neg_mask).sum(dim=1)  # (N,)
        
        #
        valid_anchor_pos = neg_mask.sum(dim=1)>0 # (N,)
        if torch.any(valid_anchor_pos):
            eps = 1e-10 # to avoid nan
            pos = pos[valid_anchor_pos]
            neg = neg[valid_anchor_pos]
            loss = -torch.log(torch.clip(pos / torch.clip(pos + neg,eps),eps )).mean()
            return loss
        else:
            return torch.tensor(0.0).to('cuda')
    
    def sample_pos_neg(ious, positive_overlap):
        pos_mask = ious > positive_overlap # (N,M)
        # neg_mask = ious < positive_overlap # (N,M)
        
        valid = (pos_mask.sum(dim=1))>0 # (N,), bool
        if valid.sum()<1: return None,None
        pos1 = torch.nonzero(valid).squeeze() # (npos,)
        pos2 = torch.argmax(ious,dim=1)[valid] # (npos,)
        
        return pos1, pos2
        
    loss = torch.tensor(0.0).to('cuda')
    src_labels_list = data_dict['src_graph']['labels']
    ref_labels_list = data_dict['ref_graph']['labels']
    src_label_masks = data_dict['src_graph']['scene_mask'].tolist()
    ref_label_masks = data_dict['ref_graph']['scene_mask'].tolist()

    for scene_id in torch.arange(data_dict['batch_size']):
        src_feats = src_stack_feats[data_dict['src_graph']['batch'][scene_id]:data_dict['src_graph']['batch'][scene_id+1]]
        ref_feats = ref_stack_feats[data_dict['ref_graph']['batch'][scene_id]:data_dict['ref_graph']['batch'][scene_id+1]]

        # feats_dist = (src_feats @ ref_feats.T) / temp # (N_i,M_i)
        ious = data_dict['instance_ious'][scene_id]
        src_labels = [src_labels_list[i] for i,m in enumerate(src_label_masks) if m==scene_id] # list
        ref_labels = [ref_labels_list[i] for i,m in enumerate(ref_label_masks) if m==scene_id] # list
        neg_mask  = ious < 0.01
        
        assert src_feats.shape[0] == ious.shape[0] and ref_feats.shape[0] == ious.shape[1]
        assert src_feats.shape[0] == len(src_labels) and ref_feats.shape[0] == len(ref_labels)
        
        pos1, pos2 = sample_pos_neg(ious, positive_overlap)
        if pos1 is None: continue
        if pos1.ndim==0: pos1 = pos1.unsqueeze(0)
        if pos2.ndim==0: pos2 = pos2.unsqueeze(0)
        
        pos1_labels = [src_labels[i] for i in pos1]
        pos2_labels = [ref_labels[i] for i in pos2]
        contrastive_loss0 = run(src_feats[pos1],ref_feats[pos2].detach(),ref_feats.detach(),neg_mask[pos1],pos1_labels,ref_labels,temp)
        contrastive_loss1 = run(ref_feats[pos2],src_feats[pos1].detach(),src_feats.detach(),neg_mask.T[pos2],pos2_labels,src_labels,temp)
        loss = loss + contrastive_loss0 + contrastive_loss1
        
    return loss

def calculate_nll_loss(logmax_scores,gt):
    
    def compute_gt_weight(logmax_scores,gt):
        '''
        positive are 1. others are all zero.
        '''
        m,n = logmax_scores.shape
        m -= 1
        n -= 1

        matches0 = gt[:,0]
        matches1 = gt[:,1]
        neg0 = torch.ones(m).to('cuda')
        neg1 = torch.ones(n).to('cuda')
        neg0[matches0] = 0
        neg1[matches1] = 0
        
        gt_weights = torch.zeros_like(logmax_scores) # (m+1,n+1)
        gt_weights[matches0,matches1] = 1   # true pos
        gt_weights[:m,-1] = neg0 # true neg
        gt_weights[-1,:n] = neg1
        
        return gt_weights
    
    # def forward(self,logmax_scores,gt):
    m, n = logmax_scores.shape
    m -= 1
    n -= 1
    if gt.shape[0]<1:
        return torch.tensor(0.0).to('cuda'), torch.tensor(0.0).to('cuda'), torch.tensor(0.0).to('cuda'), torch.tensor(0.0).to('cuda')
    assert gt[:,0].max() <= m and gt[:,1].max() <= n, 'gt match out of instance range'
    
    gt_weights = compute_gt_weight(logmax_scores,gt)
    num_pos = gt_weights[:m, :n].sum((-1, -2)).clamp(min=1.0)
    # print('find {} positive pairs'.format(num_pos))
    
    loss_sc = logmax_scores * gt_weights
    nll_loss = -loss_sc[:m, :n].sum((-1, -2))
    nll_loss /= num_pos.clamp(min=1.0)
    
    num_neg0 = gt_weights[:m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = gt_weights[-1, :n].sum(-1).clamp(min=1.0)
    nll_neg0 = -loss_sc[:m, -1].sum(-1) 
    nll_neg1 = -loss_sc[-1, :n].sum(-1)
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_loss, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0

def calculate_nll_loss_v2(logscores,ious,min_iou=0.05,alpha=2.0):     
    def compute_gt_weight(logscores,ious,alpha):
        ''' Only instances without any valid iou is treated as negative pairs.'''
        m,n = logscores.shape
        m -= 1
        n -= 1

        matches0 = ious.sum(1)>1e-3 # (m,)
        matches1 = ious.sum(0)>1e-3 # (n,)
        neg0 = torch.ones(m).to('cuda')
        neg1 = torch.ones(n).to('cuda')
        neg0[matches0] = 0
        neg1[matches1] = 0
        
        gt_weights = torch.zeros_like(logscores) # (m+1,n+1)
        gt_weights[:m,:n] = alpha * ious # (1.0,e]
        gt_weights[:m,-1] = neg0 # true neg are set to 1
        gt_weights[-1,:n] = neg1
        
        return gt_weights
    m,n = logscores.shape
    m -= 1
    n -= 1
    
    ious = torch.clamp(ious,min=0.0,max=1.0)
    ious[ious<min_iou] = 0.0
    gt_weights = compute_gt_weight(logscores,ious,alpha)
    num_pos = (gt_weights[:m,:n]>1e-3).sum((-1, -2)).clamp(min=1.0)

    loss_sc = logscores * gt_weights
    nll_loss = -loss_sc[:m, :n].sum((-1, -2))
    nll_loss /= num_pos.clamp(min=1.0)
    
    num_neg0 = gt_weights[:m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = gt_weights[-1, :n].sum(-1).clamp(min=1.0)
    nll_neg0 = -loss_sc[:m, -1].sum(-1)
    nll_neg1 = -loss_sc[-1, :n].sum(-1)
    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)
    
    return nll_loss, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.instance_match.positive_margin,
            cfg.instance_match.negative_margin,
            cfg.instance_match.positive_optimal,
            cfg.instance_match.negative_optimal,
            cfg.instance_match.log_scale,
        )
        self.positive_overlap = cfg.instance_match.positive_overlap

    def forward(self, feat_dists, overlaps):
        # loss = torch.tensor(0.0)
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
                
        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss

class FineMatchingLoss(nn.Module):
    def __init__(self,positive_radius):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = positive_radius

    def calculate_loss(self, matching_scores, gt_masks):
        '''
        Inputs:
        - matching_scores: Tensor of shape (B, N+1, M+1) giving the matching scores
        - gt_masks: Tensor of shape (B, N, M) giving the ground-truth matching masks
        '''
        assert False, 'abandoned'
        slack_row_labels = torch.eq(gt_masks.sum(2), 0) # (B, N)
        slack_col_labels = torch.eq(gt_masks.sum(1), 0) # (B, M)


        labels = torch.zeros_like(matching_scores, dtype=torch.bool)
        labels[:, :-1, :-1] = gt_masks
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()

        return loss

    def forward(self, data_dict, transforms):
        '''
        Input:
            - matching_ptr: (M,)
            - ref_matching_points: (M,K,3)
            - src_matching_points: (M,K,3)
            - matching_scores: (M,K+1,K+1)
            - transforms: (B,4,4)
        '''
        matching_ptr = data_dict['matching_ptr']
        ref_instance_pts = data_dict['ref_matching_points']
        src_instance_pts = data_dict['src_matching_points']
        matching_scores = data_dict['matching_scores']
        assert matching_scores.shape[1] == ref_instance_pts.shape[1]+1, \
            'Enable dustbin in the optimal transport of point matching'
        ref_instance_pts_mask = ref_instance_pts.sum(dim=2)>1e-3 #  data_dict['ref_instance_pts_mask'] #(M,K)
        src_instance_pts_mask = src_instance_pts.sum(dim=2)>1e-3 # data_dict['src_instance_pts_mask'] #(M,K)
        
        src_transforms = transforms[matching_ptr,:,:]
        src_aligned_instance_pts = apply_transform(src_instance_pts,src_transforms)
        assert src_transforms.shape[0] == ref_instance_pts.shape[0]
        
        dist = pairwise_distance(ref_instance_pts,src_aligned_instance_pts) # (M,K,K)
        assert dist.shape[0]==ref_instance_pts.shape[0] and \
            dist.shape[1]==ref_instance_pts.shape[1]
        corrs_masks = torch.logical_and(ref_instance_pts_mask.unsqueeze(2), 
                                        src_instance_pts_mask.unsqueeze(1)) # (M,K,K)
        gt_corr_map = torch.gt(dist,self.positive_radius**2)
        gt_corr_map = torch.logical_and(gt_corr_map, corrs_masks) # (M,K,K)
        
        slack_row_labels = torch.logical_and(torch.eq(gt_corr_map.sum(2), 0), 
                                             ref_instance_pts_mask) # (M,K)
        slack_col_labels = torch.logical_and(torch.eq(gt_corr_map.sum(1), 0), 
                                             src_instance_pts_mask) # (M,K)

        labels = torch.zeros_like(matching_scores, dtype=torch.bool) # (M,K,K)
        labels[:, :-1, :-1] = gt_corr_map
        labels[:, :-1, -1] = slack_row_labels
        labels[:, -1, :-1] = slack_col_labels

        loss = -matching_scores[labels].mean()
    
        return loss
    
class InstanceMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(InstanceMatchingLoss, self).__init__()
        # self.weighted_loss = CoarseMatchingLoss(cfg)
        self.loss_func = cfg.instance_match.loss_func
        self.gt_alpha = cfg.instance_match.gt_alpha
        self.nll_negative_weight = cfg.instance_match.nll_negative
        
        assert self.loss_func!='overlap', 'overlap loss is not supported anymore'

    def forward(self, match_output_dict, data_dict):
        
        match_gt_dict = data_dict['instance_matches'] # list
        gt_iou = data_dict['instance_ious'] # list
        B = len(match_output_dict['logmax_scores'])
        match_loss = torch.tensor(0.0).to('cuda')
        nll_pos = torch.tensor(0.0).to('cuda')
        nll_neg = torch.tensor(0.0).to('cuda')
        assert len(match_gt_dict) == B

        for scene_id in torch.arange(B):
            logmax_scores = match_output_dict['logmax_scores'][scene_id]
            # feat_dist = match_output_dict['dist'][scene_id]
            gt = match_gt_dict[scene_id]
            if gt_iou[scene_id] is None:
                nll_pos += torch.tensor(0.0).to('cuda')
                nll_neg += torch.tensor(0.0).to('cuda')
                continue
            
            if self.loss_func=='nll':
                nll_pos_, nll_neg_, num_pos, num_neg = calculate_nll_loss(logmax_scores,gt)
                nll_pos += nll_pos_
                nll_neg += nll_neg_
            elif self.loss_func=='nllv2':
                nll_pos_, nll_neg_, num_pos, num_neg = calculate_nll_loss_v2(
                    logmax_scores,gt_iou[scene_id].clone(),0.05,self.gt_alpha)
                nll_pos += nll_pos_
                nll_neg += nll_neg_
                
        if self.loss_func =='overlap':
            # return match_loss
            loss_dict = {'overlap_aware_loss':match_loss.clone().detach().item()}
        elif self.loss_func=='nll' or self.loss_func=='nllv2':
            nll_pos /= torch.tensor(B).to('cuda')
            nll_neg /= torch.tensor(B).to('cuda')
            match_loss = nll_pos + self.nll_negative_weight * nll_neg
            # if self.nll_negative_weight>0.0:
            #     match_loss += nll_neg
            loss_dict = {'nll_pos':nll_pos.clone().detach().item(), 
                         'nll_neg':nll_neg.clone().detach().item()}
            
        return match_loss, loss_dict
