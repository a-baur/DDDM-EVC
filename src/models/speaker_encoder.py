"""
Speaker encoder module, based on the
Mel-Style Encoder from Meta-StyleSpeech.

Implementation based on
DDDM-VC: https://github.com/hayeong0/DDDM-VC/blob/7f826a366b2941c7f020de07956bf5161c4979b4/model/styleencoder.py
HierSpeechpp: https://github.com/sh-lee-prml/HierSpeechpp/blob/main/styleencoder.py

Modifications:
- moved temporal_avg_pool outside the class
- added type hints and docstrings
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.modules import MultiHeadAttention
from src.util import temporal_avg_pool


class Mish(nn.Module):
    """
    Mish activation function.

    f(x) = x * tanh(softplus(x))

    Reference:
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
    """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class Conv1dGLU(nn.Module):
    """
    Conv1D + GLU (Gated Linear Unit) with residual connection.

    h(X) = (X * W + b) * sigmoid(X * V + c) + x

    Reference:
    Language Modeling with Gated Convolutional Networks
    https://arxiv.org/abs/1612.08083
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size=kernel_size, padding=2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class StyleEncoder(nn.Module):
    """
    Style encoder module.

    Based one Mel-Style Encoder from Meta-StyleSpeech.

    Description:

    1.) Spectral Processing:
        Embedding the input mel-spectrogram
        into a hidden representation.

    2.) Temporal Processing:
        Capture sequential information
        from the hidden representation.

    3.) Multi-Head Self-Attention:
        Capture global information from
        the hidden representation.

    4.) Temporal Average Pooling:
        Aggregate the hidden representation
         into a single vector.

    Reference:
    Meta-StyleSpeech: Multi-Speaker Adaptive Text-to-Speech Generation
    https://arxiv.org/abs/2004.04634
    """

    def __init__(self, in_dim=513, hidden_dim=128, out_dim=256):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.n_head,
            p_dropout=self.dropout,
            proximal_bias=False,
            proximal_init=True,
        )

        self.atten_drop = nn.Dropout(self.dropout)
        self.fc = nn.Conv1d(self.hidden_dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.spectral(x) * mask
        x = self.temporal(x) * mask

        # Transformer mask for self-attention.
        # Masks padding tokens.
        # (B, T, 1) * (B, 1, T) -> (B, T, T)
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        y = self.slf_attn(x, x, attn_mask=attn_mask)
        x = x + self.atten_drop(y)

        x = self.fc(x)

        w = temporal_avg_pool(x, mask=mask)

        return w
