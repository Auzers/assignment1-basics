import torch
import math
from torch import nn
from einops import einsum
from torch import Tensor # Tensor 用于类型标注

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


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        device=None,
        dtype=None
    ):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(d_model, d_ff, **factory_kwargs)
        self.w2 = Linear(d_ff, d_model,**factory_kwargs)
        self.w3 = Linear(d_model, d_ff, **factory_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        gate = silu(self.w1(x))
        value = self.w3(x)
        return self.w2(gate * value)


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k:int, max_seq_len:int, device=None):
        super().__init__()

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)

        inv_freq = theta ** (-pair_indices /d_k)
        angles = positions[:, None] * inv_freq[None, :]

        # 另一种写法
        # p = torch.unsqueeze(positions, dim=1)  # shape: (4, 1)
        # f = torch.unsqueeze(inv_freq, dim=0)  # shape: (1, 3)

        # 自动广播
        #angles = p * f  # shape: (4, 3)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x, token_positions):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_even = x[..., 0:: 2]
        x_odd = x[..., 1:: 2]

        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        result = torch.empty_like(x)
        result[..., 0::2] = rotated_even
        result[..., 1::2] = rotated_odd
        return result.to(in_dtype)


