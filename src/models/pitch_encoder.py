import torch
import torch.nn as nn

import util
from config import VQVAEConfig, YinEncoderConfig
from modules.vqvae import Bottleneck, Encoder
from modules.yin_encoder.modules import YINTransform_


class VQVAEEncoder(nn.Module):
    """VQ-VAE Encoder module. Returns the quantized latent codes."""

    def __init__(self, cfg: VQVAEConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg.f0_encoder)
        self.vq = Bottleneck(cfg.vq)
        self.sample_rate = cfg.sample_rate

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = util.get_normalized_f0(x, self.sample_rate)
        f0_h = self.encoder(f0)
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.vq(f0_h)
        zs = [x.detach() for x in zs]

        return zs[0].detach()


class YINEncoder(nn.Module):
    """YIN pitch encoder module."""

    def __init__(self, cfg: YinEncoderConfig) -> None:
        super().__init__()
        self.yin_transform = YINTransform_(
            sample_rate=cfg.sample_rate,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            tau_max=cfg.tau_max,
            semitone_range=cfg.semitone_range,
        )
        self.scope_lower = cfg.scope_lower
        self.scope_upper = cfg.scope_lower + cfg.out_dim

    @torch.no_grad()  # type: ignore
    def forward(
        self, x: torch.Tensor, scope_shift: torch.Tensor = None
    ) -> torch.Tensor:
        yin = self.yin_transform.yingram_batch(x)

        if scope_shift is None:
            return yin[:, self.scope_lower : self.scope_upper]
        else:
            yin = [
                yin[:, self.scope_lower + shift : self.scope_upper + shift]
                for shift in scope_shift
            ]
            return torch.stack(yin)
