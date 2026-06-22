import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import list,tuple
class RMSNorm(nn.Module):
    """支持自定义维度的RMSNorm归一化层，适配序列/图像特征图"""
    def __init__(self, dim: int, eps: float = 1e-8, norm_dim: int = -1):
        super().__init__()
        self.eps = eps
        self.norm_dim = norm_dim                         
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                   
        norm = x * torch.rsqrt(x.pow(2).mean(dim=self.norm_dim, keepdim=True) + self.eps)
                                          
        shape = [1] * x.dim()
        shape[self.norm_dim] = -1                 
        weight = self.weight.reshape(shape)
        return norm * weight

class BlockAttentionResidual(nn.Module):
    """
    块注意力残差模块（论文核心）
    输入输出均为[B, C, H, W]格式的图像特征图
    """
    def __init__(
        self,
        dim: int,
        layer_index: int,
        block_size: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.layer_index = layer_index                 
        self.block_size = block_size                         
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        self.attn_res_proj = nn.Linear(dim, 1, bias=False)            
        self.attn_res_norm = RMSNorm(dim)
        self.mlp_res_proj = nn.Linear(dim, 1, bias=False)             
        self.mlp_res_norm = RMSNorm(dim)

        nn.init.zeros_(self.attn_res_proj.weight)
        nn.init.zeros_(self.mlp_res_proj.weight)

        self.attn_norm = RMSNorm(dim)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp_norm = RMSNorm(dim)
        self.mlp_fc1 = nn.Linear(dim, int(dim * mlp_ratio), bias=False)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Linear(int(dim * mlp_ratio), dim, bias=False)
        self.mlp_dropout = nn.Dropout(dropout)

    def _block_attn_res_core(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        proj: nn.Linear,
        norm: RMSNorm,
    ) -> torch.Tensor:

                       
        V = torch.stack(blocks + [partial_block], dim=0)                  
        K = norm(V)                         

        logits = torch.einsum("c, n b t c -> n b t", proj.weight.squeeze(), K)               
        attn_weights = logits.softmax(dim=0)                 

        return torch.einsum("n b t, n b t c -> b t c", attn_weights, V)

    def forward(
        self,
        x: torch.Tensor,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor]:

        B, C, H, W = x.shape
        T = H * W                 

        x_seq = x.flatten(2).transpose(1, 2)

        if partial_block is None:
            partial_block = x_seq

        h = self._block_attn_res_core(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        h_norm = self.attn_norm(h)
        qkv = self.qkv_proj(h_norm).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_out = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        attn_out = self.attn_out_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)

        partial_block = partial_block + attn_out

        h = self._block_attn_res_core(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        h_norm = self.mlp_norm(h)
        mlp_out = self.mlp_fc1(h_norm)
        mlp_out = self.mlp_act(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out)
        mlp_out = self.mlp_fc2(mlp_out)
        mlp_out = self.mlp_dropout(mlp_out)

        partial_block = partial_block + mlp_out

        updated_blocks = blocks.copy()
        updated_partial_block = partial_block
        if (self.layer_index + 1) % self.block_size == 0:
            updated_blocks.append(partial_block)
            updated_partial_block = None

        output = h.transpose(1, 2).reshape(B, C, H, W)
        return updated_blocks, updated_partial_block, output