from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

@dataclass
class ModelOutput:
    logits = torch.Tensor
    loss: torch.Tensor | None
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None

def BuildRopeCache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute rotary position cos/sin caches

    Returns (cos, sin) each of shape of sequence length and head dimensions
    theta is the base, larger theta = slower rotation, leeading to better length extrapolation.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPe, got {head_dim}")

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def RotateHalf(x: torch.Tensor) -> torch.Tensor:
    mid = x.shape[-1] // 2
    x1 = x[..., :mid]
    x2 = x[..., mid:]
    return torch.cat([-x2, x1], dim=-1)

def ApplyRoPE(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + RotateHalf(q) * sin
    k = k * cos + RotateHalf(k) * sin
    return q, k

class RmsNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight

