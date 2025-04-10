import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

# FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

def apply_cached_rotary_emb(encoding: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return (x * encoding[0]) + (rotate_half(x) * encoding[1])

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector
            x: (b,n,3)
            return (2,b,n,d/2)
        """
        projected = self.Wr(x) # (b,n,d/2)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3) # (2,b,1,n,d/2)
        emb = emb.repeat_interleave(2, dim=-1) # (2,b,1,n,d)
        return emb.squeeze(2)
        return emb.repeat_interleave(2, dim=-1)


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        self.FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
        if allow_flash and not self.FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and self.FLASH_AVAILABLE

        if self.FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        q: (B,K,D)
        '''
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if self.FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                # v_raw = v.clone()
                v = F.scaled_dot_product_attention(args[0],args[1],args[2], attn_mask=mask).to(q.dtype)
                if mask is not None:
                    valid_mask = mask.sum(-1)>0 # (n,k)
                    nan_mask = torch.isnan(v.sum(-1)) # (n,k)
                    assert nan_mask[valid_mask].sum()<1, 'nan detected in flash attention'
                return v if mask is None else v.nan_to_num()
        elif self.FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(args[0],args[1],args[2], attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor=None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        x: (b,n,d)
        encoding: (2,b,n,d)
        mask: (b,h,n,n)
        '''
        b,n,d = x.shape
        assert b==1, 'spaital gat currently only support batch size 1'
        qkv = self.Wqkv(x) # (b,n,h*d)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2) # (b,h,n,3*d)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2] # (b,h,n,d)
        if encoding is not None:
            q = apply_cached_rotary_emb(encoding, q)
            k = apply_cached_rotary_emb(encoding, k)
        if mask is not None:
            # nodes_edge_number = mask.sum(-1)
            valid_nodes = mask.sum(-1)>0 # (b,h,n)
            # valid_nodes_number = valid_nodes.sum(-1).squeeze() # (b)
            # valid_nodes_indices = torch.nonzero(valid_nodes.squeeze(1))[:,1] # (numer_valid,2),[batch_id,node_id]
            q = q[valid_nodes].view(b,self.num_heads,-1,self.head_dim).contiguous() # (b,h,n',d)
            # k = k[valid_nodes].view(b,self.num_heads,-1,self.head_dim).contiguous() # (b,h,n',d)
            # v = v[valid_nodes].view(b,self.num_heads,-1,self.head_dim).contiguous() # (b,h,n',d)
                        
            filtered_mask = mask[valid_nodes] # (b,h,n',n)
            # filtered_mask = mask[valid_nodes_indices[:,0],0,valid_nodes_indices[:,1],valid_nodes_indices[:,1].unsqueeze(1)] # (b,h,n',n')
            assert torch.all(filtered_mask.sum(-1)>0), 'filtered mask should have at least one valid node'
            while filtered_mask.ndim<4:
                filtered_mask = filtered_mask.unsqueeze(0)
            # if valid_number.sum()<b*n:
            #     print('filter invalid graph nodes')
            
            context = torch.zeros(b,self.num_heads,n,self.head_dim).to(x.device) # (b,h,n,d)
            context[valid_nodes]= self.inner_attn(q, k, v, mask=filtered_mask) # (b,h,n',d)
        else:
            context = self.inner_attn(q, k, v, mask=mask) # (b,h,n,d)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2)) # (b,n,d)

        return x + self.ffn(torch.cat([x, message], -1))

class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
        if flash and self.FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim, heads, position_encoding, all_self_edges) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.all_self_edges = all_self_edges
        self.position_encoding = position_encoding
        self.posenc = LearnableFourierPositionalEncoding(3, None, self.head_dim)
        self.self_attn = SelfBlock(self.embed_dim, self.heads, True)
    
    
    def forward(self, x: torch.Tensor, pos:torch.Tensor, edge_index:torch.Tensor, graph_batch:torch.Tensor) -> torch.Tensor:
        '''
        x: (b,n,d)
        pos: (b,n,3)
        edge: (b,2,e)
        mask: (b,n,n)
        '''
        if x.ndim==2:
            x = x.unsqueeze(0)
            pos = pos.unsqueeze(0)
            edge_index = edge_index.unsqueeze(0)
        if torch.isnan(x).any():
            assert False, 'x nan detected'
            
        b,n,d = x.shape
        e = edge_index.shape[-1]
        if self.all_self_edges:
            mask = torch.zeros((b,self.heads,graph_batch[-1],graph_batch[-1])).bool().to(x.device)
            B_ = graph_batch.shape[0]-1
            for batch_id in torch.arange(B_):
                start_id = graph_batch[batch_id]
                stop_id = graph_batch[batch_id+1]
                mask[start_id:stop_id,start_id:stop_id] = True
        else: # set mask at edge index to 1
            mask = torch.zeros(b,self.heads,n,n).bool().to(x.device)
            mask[torch.arange(b).repeat((e)),torch.arange(self.heads).repeat((e)),edge_index[:,0,:],edge_index[:,1,:]] = True
            # print('mask shape',mask.shape)
        
        if self.position_encoding:
            encoding = self.posenc(pos) # (2,b,n,d)
        else: 
            encoding = None
        out = self.self_attn(x, encoding, mask) # (b,n,d)
        out = out.squeeze(0)
        
        # check nan
        if torch.isnan(out).any():
            assert False, 'nan detected in self-gat outupt'
        
        return out

    
if __name__=='__main__':

    print('test the self attention block')
    # B = 1
    N = 8
    heads = 1
    embed_dim = 128
    
    x = torch.randn(N,embed_dim).float()
    pos = torch.randn(N,3).float()
    edge_index = torch.randint(0, N, (2, 10))
    x = x.cuda()
    pos = pos.cuda()
    edge_index = edge_index.cuda()
    
    # print(edge_index)
    
    stransformer = SpatialTransformer(embed_dim=embed_dim,heads=heads,all_self_edges=False,position_encoding=False)
    stransformer = stransformer.cuda()
    
    out = stransformer(x,pos,edge_index,None)
    print(out.shape)
    
    exit(0)
    self_block = SelfBlock(embed_dim, heads, True)
    out = self_block(x)
    print(out.shape)
    



