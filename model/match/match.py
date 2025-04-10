import torch 
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.match.learnable_sinkhorn import LearnableLogOptimalTransport

def calculate_similarity_scores(data_dict:dict, src_feats:torch.Tensor, ref_feats:torch.Tensor):
    def run(desc0:torch.Tensor, desc1:torch.Tensor):
        '''
        desc0, desc1: (m,d)
        return: (m,n)
        '''
        _, d = desc0.shape
        mdesc0, mdesc1 = desc0 / d**0.25, desc1 / d**0.25 #
        scores = torch.einsum('md,nd->mn', mdesc0, mdesc1)
        return scores
    
    batch_size = data_dict['batch_size']
    batch_scores = []
    for scene_id in torch.arange(batch_size):
        scores = run(src_feats[data_dict['src_graph']['scene_mask']==scene_id], 
                     ref_feats[data_dict['ref_graph']['scene_mask']==scene_id])
        batch_scores.append(scores)
    return batch_scores

@torch.jit.script
def compute_correspondence_matrix(score_mat, topk, threshold): #, inclue_dustbin=False):
    r"""
    Compute matching matrix and score matrix for each patch correspondence.
    Input:
    - score_mat (Tensor): (B, K, K), or (B, K+1, K+1) if it uses dustbin.
    """

    batch_size, ref_length, src_length = score_mat.shape
    
    batch_indices = torch.arange(batch_size).cuda()

    # correspondences from reference side
    ref_topk_scores, ref_topk_indices = score_mat.topk(k=topk, dim=2)  # (B, N, K)
    ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, topk)  # (B, N, K)
    ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, topk)  # (B, N, K)
    ref_score_mat = torch.zeros_like(score_mat)
    ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
    ref_corr_mat = torch.gt(ref_score_mat, threshold)

    # correspondences from source side
    src_topk_scores, src_topk_indices = score_mat.topk(k=topk, dim=1)  # (B, K, N)
    src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, topk, src_length)  # (B, K, N)
    src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, topk, -1)  # (B, K, N)
    src_score_mat = torch.zeros_like(score_mat)
    src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
    src_corr_mat = torch.gt(src_score_mat, threshold)

    # merge results from two sides
    corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)

    # if self.use_dustbin:
    # corr_mat = corr_mat[:, :-1, :-1]
    
    return corr_mat

def simple_graph_matching_fn(data_dict:dict, stack_score_list:list, min_score:float, k:int):
    stack_logmax_scores = []
    # stack_corr_mat = []
    stack_pred_pairs = []
    
    for scene_id in torch.arange(data_dict['batch_size']):
        scores = stack_score_list[scene_id] # (m,n)
        # m,n = scores.shape
        logmax_scores0 = F.log_softmax(scores, dim=1) # (m,n)
        logmax_scores1 = F.log_softmax(scores.transpose(-1,-2).contiguous(), dim=1).transpose(-1,-2) # (m,n)
       
        logmax_scores = logmax_scores0 + logmax_scores1 # (m,n)
        scores = torch.exp(logmax_scores)
        assigment_mat = compute_correspondence_matrix(scores,k,min_score)
        
        pred_pairs = torch.nonzero(assigment_mat) # (e,2)
        stack_pred_pairs.append(torch.cat([scene_id*torch.ones(pred_pairs.shape[0],1).to(pred_pairs.device),pred_pairs],dim=1))
        stack_logmax_scores.append(scores)
    
    #
    stack_pred_pairs = torch.cat(stack_pred_pairs,dim=0)
    
    return {'pred_nodes':stack_pred_pairs.long(), 
            'scores':stack_score_list,
            'logmax_scores':stack_logmax_scores}
    
@torch.jit.script
def match_asignment_script_fn(desc0:torch.Tensor, desc1:torch.Tensor):
    b,m,d = desc0.shape
    _,n,_ = desc1.shape
    mdesc0, mdesc1 = desc0 / d**0.25, desc1 / d**0.25 #
    
    z0 = 0.1*torch.ones(b,m).to(desc0.device)
    z1 = 0.1*torch.ones(b,n).to(desc0.device)
    Kn = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1) # (m,n), nodes affinity
    
    scores0 = F.log_softmax(Kn, 2)
    scores1 = F.log_softmax(Kn.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = Kn.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1
    
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    
    scores = scores.exp()
        
    # select the top-k cells in each row and column
    batch_indices = torch.arange(b).cuda()
    threshold = 0.3
    
    src_topk_scores, src_topk_indices = scores[:, :-1, :-1].topk(3, dim=2) # (b,m,k)
    src_batch_indices = batch_indices.view(b,1,1).expand(-1,m,3) # (b,m,k)
    src_indices = torch.arange(m).view(1,m,1).expand(b,-1,3)
    src_scores_mat = torch.zeros_like(scores[:, :-1, :-1])
    src_scores_mat[src_batch_indices, src_indices, src_topk_indices] = src_topk_scores
    src_corr_mat = torch.gt(src_scores_mat, threshold) # (b,m,n)
    
    return scores

class MatchAssignment(nn.Module):
    '''
    This is a module adopted from LightGlue.
    It can be used:
        scores, sim = MatchAssignment(dim)(desc0, desc1), 
        desc0, desc1: (b,m,d)
        scores: (b,m+1,n+1), sim: (b,m,n)
    '''
    def __init__(self, dim: int, threshold: float, topk: int, multiply_matchability: bool) -> None:
        super().__init__()
        self.dim = dim
        self.threshold = threshold
        self.topk = topk
        self.multiply_matchability = multiply_matchability

        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)
    
    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25 # 
        Kn = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1) # (b,m,n), nodes affinity
        
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        logscores = self.sigmoid_log_double_softmax(Kn, z0, z1) # (b,m+1,n+1)
        scores = logscores.exp()   # between [0,1]
        # assignment = self.find_assignment(scores,threshold=self.threshold) # (b,m,n)        
        k_assignment = self.find_k_assignment(scores,threshold=self.threshold) # (b,m,n)
        return Kn.squeeze(), logscores.squeeze(), k_assignment.squeeze()
    
    def sigmoid_log_double_softmax(self,sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """create the log assignment matrix from logits and similarity"""
        b, m, n = sim.shape
        certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
        scores0 = F.log_softmax(sim, 2)
        scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
        scores = sim.new_full((b, m + 1, n + 1), 0)
        scores[:, :m, :n] = scores0 + scores1
        if self.multiply_matchability:
            scores[:, :m, :n] = scores[:, :m, :n] + certainties
        scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
        scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
        return scores

    def find_assignment(self,scores:torch.Tensor,threshold:float):
        """convert log assignment matrix to assignment matrix"""
        b, m, n = scores.shape
        m -= 1
        n -= 1
        assignment_matrix = torch.zeros_like(scores[:, :-1, :-1])
        pred_row = torch.zeros_like(scores[:, :-1, :-1])
        pred_col = torch.zeros_like(scores[:, :-1, :-1])
        valid = scores>threshold
        
        # select cells that are row-wise and col-wise maximum
        row_max = torch.argmax(scores[:, :-1, :-1], dim=2) # (b,m)
        col_max = torch.argmax(scores[:, :-1, :-1], dim=1) # (b,n)
        pred_row[torch.arange(b).unsqueeze(1), torch.arange(m).unsqueeze(0), row_max] = 1
        pred_col[torch.arange(b).unsqueeze(1), col_max, torch.arange(n).unsqueeze(0)] = 1
        assignment_matrix = pred_row * pred_col
        assignment_matrix = assignment_matrix * valid[:, :-1, :-1]
        
        
        return assignment_matrix.to(torch.int32)
    
    def find_k_assignment(self,scores:torch.Tensor,threshold:float):
        b,m,n = scores.shape
        m -= 1
        n -= 1
        batch_indices = torch.arange(b).to(scores.device)
        
        # select the top-k cells in each row
        src_topk_scores, src_topk_indices = scores[:, :-1, :-1].topk(self.topk, dim=2) # (b,m,k)
        src_batch_indices = batch_indices.view(b,1,1).expand(-1,m,self.topk) # (b,m,k)
        src_indices = torch.arange(m).view(1,m,1).expand(b,-1,self.topk)
        src_scores_mat = torch.zeros_like(scores[:, :-1, :-1])
        src_scores_mat[src_batch_indices, src_indices, src_topk_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_scores_mat, threshold) # (b,m,n)
        
        # select the top-k cells in each column
        tar_topk_scores, tar_topk_indices = scores[:, :-1, :-1].topk(self.topk, dim=1) # (b,k,n)
        tar_batch_indices = batch_indices.view(b,1,1).expand(-1,self.topk,n)
        tar_indices = torch.arange(n).view(1,1,n).expand(b,self.topk,n)
        tar_scores_mat = torch.zeros_like(scores[:, :-1, :-1])
        tar_scores_mat[tar_batch_indices, tar_topk_indices, tar_indices] = tar_topk_scores
        tar_corr_mat = torch.gt(tar_scores_mat, threshold) # (b,m,n)
        
        # # abandon
        # src_topk_scores, src_topk_indices = scores[:, :-1, :-1].topk(self.topk, dim=2) # (b,m,k)
        # src_batch_indices = batch_indices.reshape(b,1,1).expand(-1,m,self.topk) # (b,m,k)
        # src_indices = torch.arange(m).reshape(1,m,1).expand(b,-1,self.topk) # (b,m,k)
        # src_scores_mat = scores[:, :-1, :-1].clone()
        # src_scores_masks = torch.zeros_like(scores[:, :-1, :-1]).bool()
        # src_scores_masks[src_batch_indices, src_indices, src_topk_indices] = True
        # src_scores_mat[~src_scores_masks] = 0.0
        # src_corr_mat = torch.gt(src_scores_mat, threshold) # (b,m,n)
        
        corr_mat = torch.logical_and(src_corr_mat, tar_corr_mat) # (b,m,n)
        
        return corr_mat
        
        
class SinkhornMatch(nn.Module):
    def __init__(self,dim:int,iterations:int,topk:int,threshold:float) -> None:
        super().__init__()
        self.optimal_transport = LearnableLogOptimalTransport(iterations)
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.topk = topk
        self.threshold = threshold

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        assert desc0.ndim==2 and desc1.ndim==2
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _,_,d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25 #
        Kn = torch.einsum("md,nd->mn", mdesc0, mdesc1) # (m,n), nodes affinity
        scores = self.optimal_transport(Kn) # (m+1,n+1)
        # scores = scores[:-1,:-1] # (m,n)
        scores = torch.exp(scores).unsqueeze(0)
        assignment = compute_correspondence_matrix(scores=scores[:-1,:-1],topk=self.topk,threshold=self.threshold) # (b,m,n)
        return Kn.squeeze(), scores.squeeze(), assignment.squeeze()

# class SinkhornMatch_Superglue(nn.Module):
#     ''' from SuperGlue'''
#     def __init__(self, conf) -> None:
#         super().__init__()
#         self.conf = conf        
#         self.register_parameter('alpha', torch.nn.Parameter(torch.tensor(1.0)))

#         # self.iters = conf['iters']
#         # self.alpha = conf['alpha']
        
#     def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
#         scores = torch.einsum('nd,md->nm', desc0, desc1) #
#         scores = scores/ self.conf.dim**.5    
#         scores = self.log_optimal_transport(scores.unsqueeze(0), alpha=torch.tensor(self.conf.alpha).to('cuda'), iters=self.conf.iters) # (1,n'+1,m'+1)
        
#         return self.find_assignment(scores)

#     def find_assignment(self,scores):
#         # Get the matches with score above "match_threshold".
#         max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
#         indices0, indices1 = max0.indices, max1.indices # (1,n'), (1,m')
#         mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0) # (1,n'), bool
#         mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1) # (1,m'), bool
#         zero = scores.new_tensor(0)
#         mscores0 = torch.where(mutual0, max0.values.exp(), zero) # (1,n'), float
#         mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
#         valid0 = mutual0 & (mscores0 > self.conf['match_threshold']) # (1,n'), bool
#         valid1 = mutual1 & valid0.gather(1, indices1)
#         indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1)) # (1,n')
#         indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1)) # (1,m')
        
#         return {'indices0':indices0,'indices1':indices1,'mscores0':mscores0,'mscores1':mscores1}

#     def log_sinkhorn_iterations(self,Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
#         """ Perform Sinkhorn Normalization in Log-space for stability"""
#         u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
#         for _ in range(iters):
#             u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
#             v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
#         return Z + u.unsqueeze(2) + v.unsqueeze(1)

#     def log_optimal_transport(self,scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
#         """ Perform Differentiable Optimal Transport in Log-space for stability"""
#         b, m, n = scores.shape
#         one = scores.new_tensor(1)
#         ms, ns = (m*one).to(scores), (n*one).to(scores)

#         bins0 = alpha.expand(b, m, 1)
#         bins1 = alpha.expand(b, 1, n)
#         alpha = alpha.expand(b, 1, 1)

#         couplings = torch.cat([torch.cat([scores, bins0], -1),
#                             torch.cat([bins1, alpha], -1)], 1)

#         norm = - (ms + ns).log()
#         log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
#         log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
#         log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

#         Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
#         Z = Z - norm  # multiply probabilities by M+N
#         return Z
    
#     def arange_like(self,x, dim: int):
#         return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
