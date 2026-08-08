from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class ModelOutput:
    logits: torch.Tensor
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

def BuildCausalMask(
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Builds a boolean "who is allowed to look at which token" mask.

    Once a KV cache is in play, the matrix is lopsided: we might have 1 new query
    row and 300 cached key columns. PyTorch's `is_causal` flag anchors its
    triangle to the top-left of the matrix, so on a 1x300 matrix it would let our
    new token see only position 0. So we spell the mask out.
    """
    past_len = kv_len - q_len
    q_pos = torch.arange(q_len, device=device) + past_len
    k_pos = torch.arange(kv_len, device=device)
    allowed = k_pos[None, :] <= q_pos[:, None]
    return allowed[None, None, :, :]

class CausalSelfAttention(nn.Module):
    """
    Multi head causal self attention with rotary position embeddings.

    Three words, three ideas:
        "self" -- queries, keys and values all come from the same sequence.
        "causal" -- token i may only look at positions <= i, so the model can never
            cheat by reading the future it is being asked to predict.
        "multi-head" -- we slice d_model into n_heads independent subspaces and run
            attention separately in each, letting heads specialize

    Config uses n_kv_heads == n_heads, so this is plain MHA
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        ## Q, K, V each map d_model -> d_model. We reshape that output later
        # into n_heads chunks of head_dim
        # divide evenly by n_heads (config.validate enforces this)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        ## Mixes the per-head results back into a single d_model vector
        # w/o this the heads would never talk to each other
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """
        x -- (batch, seq_len, d_model)
        cos -- (seq_len, head_dim)

        Returns (output, present) where present is the (k, v) to feed back in on
        the next decoding step
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = ApplyRoPE(q, k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v) if use_cache else None

        kv_len = k.shape[2]
        if kv_len == T:
            attn_mask = None
            is_causal = True
        else:
            attn_mask = BuildCausalMask(T, kv_len, x.device)
            is_causal = False

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.resid_dropout(self.o_proj(out))
        return out, present
