import random
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.diffusion.modules import SinusoidalPosEmb


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: split last dim and rotate."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, seq_dim: int = -2
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to q and k.

    Args:
        q, k: tensors of shape (..., T, d)
        seq_dim: dimension where sequence length T resides (default -2 assuming (..., T, d))
    """
    seq_len = q.shape[seq_dim]
    head_dim = q.shape[-1]
    # Generate inverse frequency
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, device=q.device, dtype=q.dtype) / head_dim)
    )
    # Positions
    t = torch.arange(seq_len, device=q.device, dtype=q.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (T, head_dim/2)
    # Embed
    emb = torch.cat((freqs, freqs), dim=-1)  # (T, head_dim)
    cos, sin = emb.cos()[None, None, ...], emb.sin()[None, None, ...]  # (1,1,T,d)
    # Ensure dtype alignment
    cos, sin = cos.to(dtype=q.dtype), sin.to(dtype=q.dtype)
    # q, k are (..., T, d)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out


class FlashMultiHeadAttention(nn.Module):
    """Multi‑Query Attention with FlashAttention‑2 via scaled_dot_product_attention and RoPE.

    Args:
        channels:  input channel dimension
        out_channels: output dimension
        n_heads: number of *query* heads (Hq). K/V are shared (MQA) unless n_kv_heads is set >1.
        p_dropout: dropout probability applied inside attention
        n_kv_heads: number of key/value heads (defaults to 1 → MQA).
        causal: whether to use causal mask
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        n_kv_heads: int = 1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = channels // n_heads
        self.p_dropout = p_dropout
        self.causal = causal

        # Projection layers
        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv1d(
            channels, self.head_dim * n_kv_heads, kernel_size=1, bias=True
        )
        self.v_proj = nn.Conv1d(
            channels, self.head_dim * n_kv_heads, kernel_size=1, bias=True
        )
        self.out_proj = nn.Conv1d(channels, out_channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(p_dropout)

        # Weight init
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: [B, C, T_q]
            key_value: [B, C, T_k] (for self‑attention pass same as query)
            attn_mask: optional boolean mask broadcastable to (B, 1, T_q, T_k) with 0 indicating masked positions.
        Returns:
            output: [B, out_channels, T_q]
        """
        B, C, T_q = query.shape
        T_k = key_value.shape[2]

        # Project to Q/K/V
        q = self.q_proj(query)  # [B, C, T_q]
        k = self.k_proj(key_value)  # [B, H_k*d, T_k]
        v = self.v_proj(key_value)

        # Reshape for SDPA: [B, H, T, d]
        q = q.view(B, self.n_heads, self.head_dim, T_q).transpose(2, 3)
        k = k.view(B, self.n_kv_heads, self.head_dim, T_k).transpose(2, 3)
        v = v.view(B, self.n_kv_heads, self.head_dim, T_k).transpose(2, 3)

        # Rotary positional embedding
        q, k = apply_rope(q, k)

        # FlashAttention‑2 backend via SDPA. enable_gqa allows Hk < Hq.
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.p_dropout if self.training else 0.0,
            is_causal=self.causal,
            enable_gqa=(self.n_kv_heads < self.n_heads),
        )  # [B, Hq, T_q, d]

        # Merge heads and project out
        out = out.transpose(2, 3).contiguous().view(B, C, T_q)  # [B, C, T_q]
        out = self.out_proj(out)
        return out


class StyleAdaptiveLayerNorm(nn.Module):
    """Conditional Layer Normalization.

    Based on MixStyle:
    Zhou et al. Domain Generalization with MixStyle. ICLR 2021.

    :param in_dim: input dimension.
    :param gin_channels: global conditioning input dimension.
    :param mix: whether to use MixStyle.
    :param p: probability of using MixStyle.
    :param alpha: parameter of the Beta distribution for MixStyle.
    """

    def __init__(
        self,
        in_dim: int,
        gin_channels: int,
        mix: bool = True,
        p: float = 1.0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.mix = mix
        self.p = p

        self.beta_dist = torch.distributions.Beta(alpha, alpha)
        self.gamma_beta = nn.Linear(gin_channels, 2 * in_dim)
        self.norm = nn.LayerNorm(in_dim, elementwise_affine=False)

        self.gamma_beta.bias.data[:in_dim] = 1
        self.gamma_beta.bias.data[in_dim:] = 0

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p or (not self.training and self.mix):
            # skip mix during inference
            return x

        gamma_beta = self.gamma_beta(g.squeeze(-1)).unsqueeze(-1)  # (B, 2 * in_dim, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)  # (B, in_dim, 1)

        # MixStyle
        if self.mix:
            B = x.size(0)
            perm = torch.randperm(B)
            lmda = self.beta_dist.sample((B, 1, 1)).to(x.device)
            gamma = lmda * gamma + (1 - lmda) * gamma[perm]
            beta = lmda * beta + (1 - lmda) * beta[perm]

        # Perform Scailing and Shifting
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = gamma * x + beta
        return x


class StyleTransformerBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gin_channels: int,
        n_heads: int = 4,
        n_kv_heads: int = 1,
        dropout: float = 0.1,
        norm: Literal["saln", "mixstyle"] = "saln",
    ) -> None:
        super().__init__()

        self.self_attn = FlashMultiHeadAttention(
            channels=in_dim,
            out_channels=in_dim,
            n_heads=n_heads,
            p_dropout=0.1,
            n_kv_heads=n_kv_heads,
            causal=False,
        )
        if norm == "mixstyle":
            self.norm = StyleAdaptiveLayerNorm(
                in_dim=in_dim,
                gin_channels=gin_channels,
                mix=True,
                p=0.5,
                alpha=0.1,
            )
        else:
            self.norm = StyleAdaptiveLayerNorm(
                in_dim=in_dim,
                gin_channels=gin_channels,
                mix=False,
            )
        self.ffn = nn.Sequential(
            nn.Conv1d(in_dim, 4 * in_dim, 1),
            nn.ReLU(),
            nn.Conv1d(4 * in_dim, in_dim, 1),
        )

        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(1) * x_mask.unsqueeze(-1)  # (B, 1, T, T)

        residual = x
        x = self.norm(x, g)
        x = self.self_attn(x, x, attn_mask=attn_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = (x + residual) * x_mask

        residual = x
        x = self.norm(x, g)
        x = self.ffn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = (x + residual) * x_mask

        return x


class StyleTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gin_channels: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        norm: Literal["saln", "mixstyle"] = "saln",
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            [
                StyleTransformerBlock(
                    in_dim=in_dim,
                    gin_channels=gin_channels,
                    n_heads=n_heads,
                    n_kv_heads=1,
                    dropout=dropout,
                    norm=norm,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(in_dim)
        if use_positional_encoding:
            self.positional_encoding = SinusoidalPosEmb(dim=in_dim)
            self.pos_alpha = nn.Parameter(torch.Tensor([1]))
        else:
            self.positional_encoding = None  # type: ignore
            self.pos_alpha = None  # type: ignore

    def forward(
        self, x: torch.Tensor, g: torch.Tensor, x_mask: torch.Tensor
    ) -> torch.Tensor:
        # Add positional encoding
        if self.positional_encoding is not None:
            time = torch.arange(x.shape[-1], device=x.device)
            pos = self.positional_encoding(time)  # T, C
            pos = pos.transpose(0, 1).unsqueeze(0)  # 1 x C x T
            x = x + pos * self.pos_alpha
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x * x_mask

        # Apply StyleTransformer layers
        for layer in self.layers:
            x = layer(x, g, x_mask)

        # Apply final layer normalization
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)

        return x * x_mask
