import torch
import torch.nn as nn

from config import HifiGANConfig
from modules.hifigan import Generator


class HifiGAN(nn.Module):
    """
    HiFi-GAN model to generate high-fidelity audio from mel-spectrogram.
    """

    def __init__(
        self,
        cfg: HifiGANConfig,
    ) -> None:
        super().__init__()
        self.dec = Generator(
            cfg.in_dim,
            cfg.resblock,
            cfg.resblock_kernel_sizes,
            cfg.resblock_dilation_sizes,
            cfg.upsample_rates,
            cfg.upsample_initial_channel,
            cfg.upsample_kernel_sizes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dec(x)
        return y

    def infer(self, x: torch.Tensor, max_len: int = None) -> torch.Tensor:
        o = self.dec(x[:, :, :max_len])
        return o
