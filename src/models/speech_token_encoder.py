import torch
import torch.nn as nn

from config import TokenEncoderConfig
from models.dddm.preprocessor import DDDMInput


class SpeechTokenConcatenator(nn.Module):
    """Create simple speech tokens using concatenation"""

    def __init__(self, cfg: TokenEncoderConfig) -> None:
        super().__init__()
        self.pitch_dim = cfg.pitch_dim
        self.content_dim = cfg.content_dim
        self.g_dim = cfg.gin_channels
        self.out_dim = cfg.out_dim

        pitch_in = cfg.pitch_dim + cfg.gin_channels
        content_in = cfg.content_dim + cfg.gin_channels

        self.emb_src = nn.Conv1d(pitch_in, self.out_dim, kernel_size=1)
        self.emb_ftr = nn.Conv1d(content_in, self.out_dim, kernel_size=1)

    @torch.no_grad()  # type: ignore
    def forward(
        self, x: DDDMInput, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g_expanded = g.expand(-1, -1, x.emb_pitch.size(-1))

        src_in = torch.cat([x.emb_pitch, g_expanded], dim=1)
        src_tkn = self.emb_src(src_in)

        ftr_in = torch.cat([x.emb_content, g_expanded], dim=1)
        ftr_tkn = self.emb_ftr(ftr_in)
        return src_tkn, ftr_tkn
