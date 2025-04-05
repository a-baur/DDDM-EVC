import numpy as np
import parselmouth
import torch
import torch.nn as nn

import util
from config import VQVAEConfig, YinEncoderConfig
from modules.vqvae import Bottleneck, Encoder
from modules.yin_encoder.yingram import Yingram


class VQVAEEncoder(nn.Module):
    """VQ-VAE Encoder module. Returns the quantized latent codes."""

    def __init__(self, cfg: VQVAEConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg.f0_encoder)
        self.vq = Bottleneck(cfg.vq)
        self.sample_rate = cfg.sample_rate

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f0 = util.get_normalized_f0(x, self.sample_rate, framework="parselmouth")
        f0_h = self.encoder(f0)
        zs, _, _, _ = self.vq(f0_h)

        return zs[0].detach()


class YINEncoder(nn.Module):
    """YIN pitch encoder module."""

    def __init__(self, cfg: YinEncoderConfig) -> None:
        super().__init__()
        self.yin_transform = Yingram(
            sample_rate=cfg.sample_rate,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            scope_fmin=cfg.scope_fmin,
            scope_fmax=cfg.scope_fmax,
            bins=cfg.bins,
        )
        self.sample_rate = cfg.sample_rate

    @torch.no_grad()  # type: ignore
    def forward(
        self, x: torch.Tensor, semitone_shift: torch.Tensor = None
    ) -> torch.Tensor:
        return self.yin_transform(x, semitone_shift)

    def yin_with_pitch_shift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the yin with pitch shift."""
        x_pitch = torch.Tensor([_extract_pitch(s, self.sample_rate) for s in x])
        t_pitch = torch.Tensor([_extract_pitch(s, self.sample_rate) for s in t])
        x_midi = 69 + torch.log2(x_pitch / 440) * 12
        t_midi = 69 + torch.log2(t_pitch / 440) * 12
        semitone_shift = t_midi - x_midi
        return self.yin_transform(x, semitone_shift)


def _extract_pitch(x: torch.Tensor, sample_rate: int) -> float:
    """Extract pitch from audio."""
    pitch = parselmouth.praat.call(
        parselmouth.Sound(x.cpu().numpy(), sampling_frequency=sample_rate),
        "To Pitch",
        0.01,
        75,
        600,
    ).selected_array["frequency"]
    return np.median(pitch[pitch > 1e-5]).item()
