import torch
import torch.nn as nn

from config import TokenEncoderConfig
from models import DDDMInput


class SpeechTokenConcatenator(nn.Module):
    """Improved simple speech tokens using normalization and small feature adaptation"""

    def __init__(self, cfg: TokenEncoderConfig) -> None:
        super().__init__()
        self.pitch_dim = cfg.pitch_dim
        self.content_dim = cfg.content_dim
        self.g_dim = cfg.gin_channels
        self.out_dim = cfg.out_dim

        # Simple feature adapters
        self.pitch_proj = nn.Conv1d(cfg.pitch_dim, cfg.pitch_dim, kernel_size=1)
        self.content_proj = nn.Conv1d(cfg.content_dim, cfg.content_dim, kernel_size=1)

        # Normalizations
        self.norm_pitch = nn.LayerNorm(cfg.pitch_dim)
        self.norm_content = nn.LayerNorm(cfg.content_dim)
        self.norm_power = nn.LayerNorm(1)

        # Embedding convolutions
        pitch_in = cfg.pitch_dim + cfg.gin_channels + 1
        content_in = cfg.content_dim + cfg.gin_channels + 1

        self.emb_src = nn.Conv1d(pitch_in, self.out_dim, kernel_size=1)
        self.emb_ftr = nn.Conv1d(content_in, self.out_dim, kernel_size=1)

    def forward(
        self, x: DDDMInput, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g_expanded = g.expand(-1, -1, x.emb_pitch.size(-1))

        # Feature adaptation
        pitch = self.pitch_proj(x.emb_pitch)
        content = self.content_proj(x.emb_content)

        # Normalization
        pitch = self.norm_pitch(pitch.transpose(1, 2)).transpose(1, 2)
        content = self.norm_content(content.transpose(1, 2)).transpose(1, 2)

        power = torch.pow(x.mel, 2).sum(dim=1, keepdim=True) + 1e-6
        power = torch.log(power)
        power = self.norm_power(power.transpose(1, 2)).transpose(1, 2)

        # Token creation
        src_in = torch.cat([pitch, g_expanded, power], dim=1)
        src_tkn = self.emb_src(src_in)

        ftr_in = torch.cat([content, g_expanded, power], dim=1)
        ftr_tkn = self.emb_ftr(ftr_in)

        return src_tkn, ftr_tkn
