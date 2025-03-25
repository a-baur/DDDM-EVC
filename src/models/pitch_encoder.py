import torch
import torch.nn as nn

import util
from config import VQVAEConfig
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

    def __init__(self) -> None:
        super().__init__()
        self.yin_transform = YINTransform_(
            sample_rate=16000,
            win_length=1480,
            hop_length=320,
            tau_max=1480,
        )
        self.scope_lower = 15
        self.scope_upper = 65

    @torch.no_grad()  # type: ignore
    def forward(
        self, x: torch.Tensor, scope_shift: torch.Tensor = None
    ) -> torch.Tensor:
        yin = self.yin_transform.yingram_batch(x)

        if scope_shift is None:
            return yin[:, self.scope_lower : self.scope_upper]
        else:
            return torch.stack(
                [
                    yin[:, self.scope_lower + shift : self.scope_upper + shift]
                    for shift in scope_shift
                ]
            )
