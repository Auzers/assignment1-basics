import torch
import math
from torch import nn
from einops import einsum
from torch import tensor

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs)) 不包含偏置项，这和大多数现代LLM是一致的
        std = math.sqrt(2.0 / (in_features + out_features))

        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std
        )
    
    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

        
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))

        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=1.0, # 这里不用和Linear一样的方法，linear不做方差约束，数值方差会随维度膨胀，embedding直接取数字，不会膨胀
            a=-3.0,
            b= 3.0 
        )
    
    def forward(self, token_ids):
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))
        self.eps = eps
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps
        )
        result = (x/rms) * self.weight
        return result.to(in_dtype)





