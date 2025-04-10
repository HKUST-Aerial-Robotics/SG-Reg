'''
A graph convolution network. 
Based on the edge, we extract corners for each node.
The node features are updated by the corners' features.
'''

import torch
import torch.nn as nn
from sgreg.gnn.spatial_attention import Attention

def GetCornerIndices(pos:torch.Tensor, 
                     edge_indices:torch.Tensor, 
                     padding:str='zero',
                     K:int=20):
    '''
    Select all the corners for each anchor point. 
    A corner should be formed by the anchor point and its two nearby points.
    Input:
        - pos: (N,3), the position of each node
        - edge_indices: (2,E), the edge indices
        - k: int
    Return:
        - anchor_indices: (N')
        - corner_indices: (N',K,2). Notice N'<=N, due to some nodes have less than 2 edges.
        - corner_masks: (N',K)
        - padding: 'zero' or 'random'
    '''

    N = pos.shape[0]
    anchor_indices = [] # [anchor_index], (N)
    corner_indices = [] # [e0_index, e1_index]
    corner_masks = [] # [mask]
    
    # Loop over each node
    for anchor_idx in torch.arange(N):
        anchor_edges_mask = edge_indices[0,:] == anchor_idx

        if anchor_edges_mask.sum() < 2:
            continue
        ref_indices = edge_indices[1, anchor_edges_mask] # (e_i)
        # assert torch.unique(ref_indices).shape[0] == ref_indices.shape[0], 'ref_indices should be unique'
        
        # Generate all possible pairs from ref_indices
        corner_indices_i = torch.combinations(ref_indices,2) # (C(e_i),2)
        corner_mask = torch.zeros(K).to(torch.bool).to(corner_indices_i.device)
        if corner_indices_i.shape[0]<K:
            corner_mask[:corner_indices_i.shape[0]] = True

            if padding=='zero':
                corner_indices_i = torch.cat(
                                [corner_indices_i,
                                 (N)*torch.ones((K-corner_indices_i.shape[0],2)).long().to(corner_indices_i.device)],
                                dim=0) # (K,2)
            elif padding=='random':
                assert False, 'Abaondoned'
                padded_corner_ids = torch.randint(corner_indices_i.shape[0],
                                                        (K-corner_indices_i.shape[0],)).to(corner_indices_i.device)
                corner_indices_i = torch.cat([corner_indices_i,
                                            corner_indices_i[padded_corner_ids]],dim=0) # (K,2)
            else:
                raise ValueError('The padding method should be either zero or random.')
        elif corner_indices_i.shape[0]>=K:
            corner_mask[:] = True
            # corner_indices_i = corner_indices_i[:K] # (K,2)
            corner_indices_i = corner_indices_i[torch.randperm(corner_indices_i.shape[0])][:K] # (K,2)
        
        assert corner_indices_i.shape[0]==K, 'The number of corners should be equal to K.'        
        
        anchor_indices.append(anchor_idx)
        corner_indices.append(corner_indices_i.unsqueeze(0))
        corner_masks.append(corner_mask.unsqueeze(0))

    if len(corner_indices)<1:
        return torch.tensor([]).to(pos.device), torch.tensor([]).to(pos.device), torch.tensor([]).to(pos.device)
        
    anchor_indices = torch.tensor(anchor_indices).to(pos.device) # (N')
    corner_indices = torch.cat(corner_indices,dim=0) # (N',K,2)
    corner_masks = torch.cat(corner_masks,dim=0) # (N',K)
    
    anchor_triplets_count = torch.sum(corner_masks,dim=1,keepdim=True).float() # (N,1)
    assert anchor_triplets_count.min() > 0, 'The denominator should be greater than 0.'
    
    return anchor_indices, corner_indices, corner_masks

def GetCornerIndicesNew(pos:torch.Tensor,
                        edge_indices:torch.Tensor,
                        K:int=20):
    
    N = pos.shape[0]
    device = pos.device
    adj_matrix = torch.zeros((N,N)).to(pos.device)
    adj_matrix[edge_indices[0],edge_indices[1]] = 1

    # anchor_indices = [] # [anchor_index], (N)
    corner_indices = [] # [anchor_index, corner0_index, corner1_index], (N,K,3)
    corner_masks = [] # [mask], (N,K)
    padding_valud = N
    
    # From the adj_matrix, generate corners for each anchor node.
    # Each corner should be formed by the anchor node and its two nearby nodes.
    # The corners of an anchor node are not directed, where each pair of neighbors should be unique.
    # If an anchor has <K, we pad the corner indices with a padding value.
    # The corner mask is used to mask the padded corners.

def GetCornerIndicesDS(pos: torch.Tensor, edge_indices: torch.Tensor, K: int = 20):
    """
    Generate by DeepSeekV3
    """
    N = pos.shape[0]
    device = pos.device

    # Create adjacency matrix
    adj_matrix = torch.zeros((N, N), dtype=torch.bool, device=device)
    adj_matrix[edge_indices[0], edge_indices[1]] = True

    # Get the number of neighbors for each node
    neighbor_counts = adj_matrix.sum(dim=1)  # (N,)

    # Generate all possible pairs of neighbors for each node
    neighbor_indices = torch.nonzero(adj_matrix)  # (E, 2), where E is the number of edges

    # Group neighbor indices by anchor node
    unique_anchors, counts = torch.unique(neighbor_indices[:, 0], return_counts=True)  # (N',), (N',)

    # Initialize corner_indices and corner_masks
    corner_indices = torch.full((N, K, 3), N, dtype=torch.long, device=device)  # (N, K, 3)
    corner_masks = torch.zeros((N, K), dtype=torch.bool, device=device)  # (N, K)

    # Iterate over unique anchor nodes
    for anchor_idx in unique_anchors:
        # Get all neighbors of the current anchor node
        neighbors = neighbor_indices[neighbor_indices[:, 0] == anchor_idx, 1]  # (num_neighbors,)

        # Generate all unique pairs of neighbors
        if neighbors.shape[0] >= 2:
            pairs = torch.combinations(neighbors, 2)  # (num_pairs, 2)
            num_pairs = pairs.shape[0]

            # Select up to K pairs
            if num_pairs > K:
                selected_indices = torch.randperm(num_pairs)[:K]  # Randomly select K pairs
                pairs = pairs[selected_indices]
                corner_masks[anchor_idx] = True  # All K pairs are valid
            else:
                corner_masks[anchor_idx, :num_pairs] = True  # Only the first num_pairs are valid

            # Fill corner_indices with [anchor_idx, pair[0], pair[1]]
            corner_indices[anchor_idx, :pairs.shape[0], 0] = anchor_idx
            corner_indices[anchor_idx, :pairs.shape[0], 1:] = pairs

    return corner_indices, corner_masks

@torch.jit.script
def position_embedding_test_fn(emb_indices):
    '''
    Input,
        - emb_indices: (N',K)
    '''
    
    d_model = 16
    div_indices = torch.arange(0, d_model, 2).float()
    div_term = torch.exp(div_indices * (-torch.log(torch.tensor(10000.0)) / d_model))
    omegas = emb_indices.reshape(-1, 1, 1) * div_term.reshape(1, -1, 1)  # (-1, d_model/2, 1)
    omegas_tmp = emb_indices.view(-1, 1, 1) * div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
    assert torch.allclose(omegas,omegas_tmp)
    
    sin_embeddings = torch.sin(omegas)
    cos_embeddings = torch.cos(omegas)
    embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
    embeddings = embeddings.reshape((-1,emb_indices.shape[1],d_model))  # (*, d_model)
    # embeddings = embeddings.view(*emb_indices.shape, d_model)  # (*, d_model)
    embeddings = embeddings.detach()
    return embeddings

@torch.jit.script
def gnn_triplet_test_fn(x,pos,anchor_indices,corner_indices,corner_masks):
    
    # 
    padded_x = torch.cat([x,torch.zeros((1,x.shape[1])).to(x.device)],dim=0) # (N+1,in_channels)
    padded_pos = torch.cat([pos,torch.zeros((1,3)).to(pos.device)],dim=0) # (N+1,3)
    x_messages = torch.zeros_like(x)
    
    # 
    triplet_anchor_points = padded_pos[anchor_indices] # (N',3)
    triplet_vector0 = (padded_pos[corner_indices[:,:,0]] - triplet_anchor_points.unsqueeze(1)).detach() # (N',K,3)
    triplet_vector1 = (padded_pos[corner_indices[:,:,1]] - triplet_anchor_points.unsqueeze(1)).detach() # (N',K,3)
    triplet_dists = torch.stack([torch.linalg.norm(triplet_vector0,dim=-1),
                                torch.linalg.norm(triplet_vector1,dim=-1)],dim=2) # (N',K,2)
    triplet_vector0[:,:,2] = 0.0
    triplet_vector1[:,:,2] = 0.0

    cos_values = torch.sum(triplet_vector0 * triplet_vector1, dim=-1) # (N',K)
    cos_values = cos_values / (torch.linalg.norm(triplet_vector0,dim=-1) * torch.linalg.norm(triplet_vector1,dim=-1)+1e-6) # (N',K)
    sin_vector = torch.cross(triplet_vector0, triplet_vector1, dim=-1)  # (N',K,3) 

    # 
    corner_indices_reorder = corner_indices.clone()
    # 3. Concatenate the corner features
    angle_features = position_embedding_test_fn(cos_values) # (N',K,hidden_channels)
    triplet_feats = torch.cat([padded_x[corner_indices_reorder[:,:,0]],padded_x[corner_indices_reorder[:,:,1]],angle_features], 
                            dim=2).contiguous() # (N',K, 2*in_channels+angle_dist_dim)
    return cos_values

class SinusoidalPositionalEmbedding(nn.Module):
    # import numpy as np
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*), (N',K) or (N',K,2)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        assert emb_indices.ndim == 2, 'only support 2 dimensions.'
        input_shape = emb_indices.shape
        input_shape_list = list(input_shape) + [self.d_model]
        omegas = emb_indices.reshape(-1, 1, 1) * self.div_term.reshape(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.reshape((-1,input_shape[1],self.d_model))  # (*, d_model)
        # embeddings = embeddings.view(input_shape_list) # (*, d_model)
        # embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings
    
class SinusoidalPositionalEmbedding2(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding2, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*), (N',K,2)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        assert emb_indices.ndim == 3, 'only support 3 dimensions.'
        input_shape = emb_indices.shape
        omegas = emb_indices.reshape(-1, 1, 1) * self.div_term.reshape(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.reshape((-1,input_shape[1],2,self.d_model))  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings

class TripletGNN(nn.Module):
    def __init__(self, in_channels, angle_dist_dim, out_channels, reduce, enable_dist_embedding=False,triplet_mlp_method='concat',triplet_activation='gelu'):
        super().__init__()
        self.enable_angle_embedding = True
        self.enable_dist_embedding = enable_dist_embedding
        self.angle_embedding = SinusoidalPositionalEmbedding(angle_dist_dim)
        self.enable_attn = True

        self.in_channels = in_channels
        self.mlp0 = nn.Linear(in_channels, out_channels)
        self.reduce = reduce
        triplet_dimension = 2 * in_channels + angle_dist_dim # concat [e0,e1,angle,dist0,dist1]
        if enable_dist_embedding:
            self.dist_embedding = SinusoidalPositionalEmbedding2(angle_dist_dim)
            triplet_dimension = triplet_dimension + 2*angle_dist_dim
        self.triplet_dimension = triplet_dimension 
        
        if self.enable_attn:
            self.Wqkv = torch.nn.Linear(triplet_dimension, 3*in_channels)
            self.attn = Attention(True)   

        self.out_projector = nn.Linear(2 * in_channels, in_channels)
        if triplet_activation=='gelu':
            activation = nn.GELU()
        elif triplet_activation=='relu':
            activation = nn.ReLU()
        else:
            raise ValueError('The activation should be either gelu or relu.')
        self.ffn = nn.Sequential(
            nn.Linear(2 * in_channels, 2 * in_channels),
            nn.LayerNorm(2 * in_channels, elementwise_affine=True),
            activation,
            nn.Linear(2 * in_channels, in_channels))     
            
        self.triplet_mlp = nn.Linear(triplet_dimension, out_channels)
        self.padding_triplets = 'zero'
        # self.triplet_attention_mask = False # Wheather to mask those padded triplet features
        self.triplet_mlp_method = triplet_mlp_method
        self.all_edges = False
    
    def forward(self, x, 
                      pos, 
                      anchor_indices, 
                      corner_indices, 
                      corner_masks):
        '''
        Input:
            - x: (N, in_channels)
            - pos: (N, 3)
            - anchor_indices: (N')
            - corner_indices: (N',K,2)
            - corner_masks: (N',K)
        '''
        # N = x.shape[0]
        
        padded_x = torch.cat([x,torch.zeros((1,x.shape[1])).to(x.device)],dim=0) # (N+1,in_channels)
        padded_pos = torch.cat([pos,torch.zeros((1,3)).to(pos.device)],dim=0) # (N+1,3)

        x_messages = torch.zeros_like(x)

        # if anchor_indices.shape[0]>0:
        triplet_anchor_points = padded_pos[anchor_indices] # (N',3)
        triplet_vector0 = (padded_pos[corner_indices[:,:,0]] - triplet_anchor_points.unsqueeze(1)).detach() # (N',K,3)
        triplet_vector1 = (padded_pos[corner_indices[:,:,1]] - triplet_anchor_points.unsqueeze(1)).detach() # (N',K,3)
        triplet_dists = torch.stack([torch.linalg.norm(triplet_vector0,dim=-1),
                                    torch.linalg.norm(triplet_vector1,dim=-1)],dim=2) # (N',K,2)
        triplet_vector0[:,:,2] = 0.0
        triplet_vector1[:,:,2] = 0.0

        cos_values = torch.sum(triplet_vector0 * triplet_vector1, dim=-1) # (N',K)
        cos_values = cos_values / (torch.linalg.norm(triplet_vector0,dim=-1) * torch.linalg.norm(triplet_vector1,dim=-1)+1e-6) # (N',K)
        sin_vector = torch.cross(triplet_vector0, triplet_vector1, dim=-1)  # (N',K,3) 

        # 2. Sort each corner indices in anticlockwise order
        reorder_corner_masks = sin_vector[:,:,2] < 0 # [0,180), (N',K)
        # todo: mind the onnx warning here
        corner_indices_reorder = corner_indices.clone()
        corner_indices_reorder[reorder_corner_masks] = corner_indices_reorder[reorder_corner_masks][:,[1,0]] # switch e0 and e1
        # corner_indices[reorder_corner_masks] = corner_indices[reorder_corner_masks][:,[1,0]] # switch e0 and e1
        triplet_dists[reorder_corner_masks] = triplet_dists[reorder_corner_masks][:,[1,0]] # switch dist0 and dist1
        assert corner_indices_reorder.ndim==3, 'The corner_indices should have 3 dimensions.'

        ## Verify
        new_vector0 = (padded_pos[corner_indices_reorder[:,:,0]] - triplet_anchor_points.unsqueeze(1)).detach() # (N',K,3)
        new_vector1 = (padded_pos[corner_indices_reorder[:,:,1]] - triplet_anchor_points.unsqueeze(1)).detach() # 
        new_vector0[:,:,2] = 0.0
        new_vector1[:,:,2] = 0.0
        new_sin_vector = torch.cross(new_vector0, new_vector1, dim=-1) # (N',K,3)
        triplet_false_masks = new_sin_vector[corner_masks][:,2] < -1e-6 # (-1)
        
        # if false_masks.sum()>0:
        #     print('{} false masks'.format(false_masks.sum()))
        #     print('false triplet sin values:', new_sin_vector[corner_masks][false_masks][:,2])
        #     assert False, 'The corners should be sorted in anticlockwise order.'

        # 3. Concatenate the corner features
        angle_features = self.angle_embedding(cos_values) # (N',K,hidden_channels)
        triplet_feats = torch.cat([padded_x[corner_indices_reorder[:,:,0]],
                                   padded_x[corner_indices_reorder[:,:,1]],
                                   angle_features], 
                                dim=-1).contiguous() # (N',K, 2*in_channels+angle_dist_dim)
    
        if self.enable_dist_embedding:            
            dist_embeddings = self.dist_embedding(triplet_dists) # (N',K,2,angle_dist_dim)
            dist_embeddings = torch.flatten(dist_embeddings, start_dim=2) # (N',K,2*angle_dist_dim)
            triplet_feats = torch.cat([triplet_feats, dist_embeddings],dim=2)
      
        triplet_qkv = self.Wqkv(triplet_feats) # (N',K,3*in_channels)
        
        triplet_qkv = torch.unflatten(triplet_qkv, -1, (3,self.in_channels)) # (N',K, 3, in_channels)
        q,k,v = triplet_qkv[:,:,0],triplet_qkv[:,:,1],triplet_qkv[:,:,2] # (N',K, in_channels)
        
        triplet_attn_masks = None
        triplet_attn_feats = self.attn(q,k,v,triplet_attn_masks) # (N',K, in_channels)
        triplets_denominator = torch.sum(corner_masks,dim=1,keepdim=True).float() # (N',1)
        # assert triplets_denominator.min() > 0, 'The denominator should be greater than 0.'
        mean_triplet_feats = torch.sum(triplet_attn_feats * corner_masks.unsqueeze(2),dim=1) / triplets_denominator # (N',in_channels)
        
        # Gather by index
        x_messages[anchor_indices] = mean_triplet_feats # (N,in_channels)
        
        # Gather by scatter
        # flatten_triplet_feats = triplet_attn_feats[corner_masks]
        # all_anchor_indices = anchor_indices.unsqueeze(1).expand(-1,corner_masks.shape[1]) # (N',K)
        # flatten_anchor_indices = all_anchor_indices[corner_masks]
        # flatten_triplet_feats_checksum = torch.abs(flatten_triplet_feats.sum(-1))>1e-6 # 
        # assert torch.all(flatten_triplet_feats_checksum), 'The triplet features should not be zero.'   
        # # assert torch.isnan(flatten_triplet_feats).sum() == 0, 'x_messages contains nan values.'
        # x_messages = x_messages.scatter_reduce(0,
        #                                         flatten_anchor_indices.unsqueeze(1).expand(-1,self.in_channels),
        #                                         flatten_triplet_feats,
        #                                         self.reduce,include_self=False) # (N, in_channels)
    
        if self.triplet_mlp_method == 'concat':
            out = torch.cat([x,x_messages],dim=1) # (N, 2*in_channels)
        elif self.triplet_mlp_method == 'ffn':
            out = x + self.ffn(torch.cat([x,x_messages],dim=1)) # (N, in_channels)
        elif self.triplet_mlp_method =='projector':
            out = x + self.out_projector(torch.cat([x,x_messages],dim=1)) # (N, in_channels)
        else:
            raise ValueError('The triplet_mlp_method should be either concat or ffn.')

        return out, triplet_false_masks

import torch

def generate_random_graph_edge_indices(N: int, E: int):
    """
    Generate edge indices for a random graph with N nodes, where each node is connected to K random nodes (excluding itself).
    Input:
        - N: int, number of nodes
        - K: int, number of edges per node
    Return:
        - edge_indices: (2, N*K), edge indices for the random graph
    """
    if E >= N:
        raise ValueError("K must be less than N, as a node cannot connect to itself.")

    edge_indices = []

    for i in range(N):
        # Create a list of all nodes except itself
        candidates = torch.cat([torch.arange(0, i), torch.arange(i + 1, N)])

        # Randomly select K nodes from the candidates
        selected = torch.randperm(len(candidates))[:E]
        connected_nodes = candidates[selected]

        # Add edges (i -> connected_nodes)
        edges = torch.stack([torch.full((E,), i), connected_nodes], dim=0)
        edge_indices.append(edges)

    # Combine all edges into a single tensor
    edge_indices = torch.cat(edge_indices, dim=1)

    return edge_indices
    
if __name__=='__main__':
    from sgreg.utils.tictoc import  TicToc
    N = 8
    E = 3
    K = 3
    x = torch.randn(N, 16)
    pos = torch.randn(N, 3) # (N,3)
    # edge_indices = torch.randint(0,N,(2,10)) # (2,E)
    # edge_indices = torch.tensor([[0, 0, 0, 1, 2, 2, 3, 3], 
    #                              [1, 2, 3, 2, 0, 1, 4, 5]])  # Example edges
    edge_indices = generate_random_graph_edge_indices(N, E)
    print('edge indices:\n', edge_indices)

    tictoc = TicToc()
    anchor_indices, corner_indices, corner_masks = GetCornerIndices(pos, 
                                                                    edge_indices,
                                                                    'zero',
                                                                    K) # [anchor_index, e0_index, e1_index]
    
    print('anchor_indices:', anchor_indices)
    # print('corner_indices:', corner_indices)
    print('corner_masks:', corner_masks)
    print('Corner indices generation takes P {:.2f}ms'.format(tictoc.toc()*1000))
    
    exit(0)
    net = TripletGNN(16, 16, 16, 'mean', True)
    
    out = net(x, pos, anchor_indices, corner_indices, corner_masks)
    
    # anchor_indices = triplet_indices[:,0] # (T)
    print('edge_indices:\n', edge_indices)
    print('anchor_indices:\n', anchor_indices)
    print('corner_indices:\n', corner_indices)
    
    exit(0)
    # mask = anchor_indices.unsqueeze(1) == anchor_indices.unsqueeze(0) # (T,T)
    # print(mask)
    # print(mask.shape)
     
    # test the sinusoidal positional embedding
    emb = SinusoidalPositionalEmbedding(16)
    emb_indices = torch.arange(0,10).float().view(2,5)
    out = emb(emb_indices)
    print(emb_indices)
    print(out.shape)