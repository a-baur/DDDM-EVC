import torch
import torch.nn as nn

from config import PitchEncoderConfig
from modules.vqvae import Bottleneck, Decoder, Encoder

LRELU_SLOPE = 0.1


class PitchEncoder(nn.Module):
    def __init__(self, cfg: PitchEncoderConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg.f0_encoder)
        self.vq = Bottleneck(cfg.vq)
        self.decoder = Decoder(cfg.f0_decoder)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0_h = self.encoder(x)
        _, f0_h_q, f0_commit_losses, f0_metrics = self.vq(f0_h)
        f0 = self.decoder(f0_h_q)

        return f0, f0_commit_losses, f0_metrics

    @torch.no_grad()  # type: ignore
    def code_extraction(self, x: torch.Tensor) -> torch.Tensor:
        f0_h = self.encoder(x)
        f0_h = [x.detach() for x in f0_h]
        zs, _, _, _ = self.vq(f0_h)
        zs = [x.detach() for x in zs]

        return zs[0].detach()
