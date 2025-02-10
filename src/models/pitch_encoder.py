import torch
import torch.nn as nn

from config import VQVAEConfig
from modules.vqvae import Bottleneck, Encoder


class VQVAEEncoder(nn.Module):
    """VQ-VAE Encoder module. Returns the quantized latent codes."""

    def __init__(self, cfg: VQVAEConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg.f0_encoder)
        self.vq = Bottleneck(cfg.vq)

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0_h = self.encoder(x)
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.vq(f0_h)
        zs = [x.detach() for x in zs]

        return zs[0].detach()
