import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FullAttnResBlock(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        self.queries = nn.Parameter(torch.randn(num_layers, dim))
        self.norm = nn.RMSNorm(dim)
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, layer_outputs: torch.Tensor, current_layer: int) -> torch.Tensor:
        batch_size = layer_outputs.shape[0]
        num_previous = layer_outputs.shape[1]
        
        q = self.queries[current_layer].unsqueeze(0)
        q = q.unsqueeze(1)
        
        k = self.norm(layer_outputs)
        
        scores = torch.sum(q * k, dim=-1)
        scores = scores / self.temperature
        attn_weights = F.softmax(scores, dim=-1)
        
        aggregated = torch.sum(attn_weights.unsqueeze(-1) * layer_outputs, dim=1)
        
        return aggregated, attn_weights

class AttnResLayer(nn.Module):
    def __init__(self, dim: int, num_layers: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn_res = FullAttnResBlock(dim, num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm = nn.RMSNorm(dim)
        
    def forward(self, layer_outputs: torch.Tensor, current_layer: int):
        aggregated, attn_weights = self.attn_res(layer_outputs, current_layer)
        output = self.mlp(self.norm(aggregated))
        
        return output, attn_weights