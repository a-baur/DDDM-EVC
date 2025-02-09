import torch
import torch.nn as nn

import util
from config import ModelsConfig

from .diffusion import Diffusion
from .source_filter_encoder import SourceFilterEncoder
from .speaker_encoder import MetaStyleSpeech


class DDDM(nn.Module):
    def __init__(self, cfg: ModelsConfig) -> None:
        super().__init__()
        self.style_encoder = MetaStyleSpeech(cfg.speaker_encoder)
        self.source_filter_encoder = SourceFilterEncoder(cfg)
        self.diffusion = Diffusion(cfg.diffusion)

    def forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_length: torch.Tensor,
        mixup: bool = False,
        sample_rate: int = 16000,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_length: Number of unpadded frames in mel-spectrgram
        :param mixup: Whether to use prior mixup or not
        :param sample_rate: Sampling rate of the input waveform
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        # Encode the input waveform into diffusion priors
        x_mask = util.sequence_mask(x_length, x_mel.size(2)).to(x_mel.dtype)
        g = self.style_encoder(x_mel, x_mask)
        src_mel, ftr_mel = self.source_filter_encoder(
            x, x_mel, x_mask, mixup, sample_rate
        )

        # Compute diffused mean
        src_mean_x, ftr_mean_x = self.diffusion.compute_diffused_mean(
            x_mel, x_mask, src_mel, ftr_mel, 1.0
        )

        # Pad the sequences for U-Net compatibility
        max_length_new = util.get_u_net_compatible_length(x_mel.size(-1))
        src_mean_x, ftr_mean_x, src_mel, ftr_mel, x_mask = util.pad_tensors_to_length(
            [src_mean_x, ftr_mean_x, src_mel, ftr_mel, x_mask], max_length_new
        )

        # Add noise to diffused mean to create priors for diffusion
        start_n = torch.randn_like(src_mean_x, device=src_mean_x.device)
        src_mean_x.add_(start_n)
        ftr_mean_x.add_(start_n)

        # Diffusion
        y_src, y_ftr = self.diffusion(
            src_mean_x, ftr_mean_x, x_mask, src_mel, ftr_mel, g, 6, "ml"
        )
        y = (y_src + y_ftr) / 2

        if return_enc_out:
            enc_out = src_mel + ftr_mel
            return y, enc_out
        else:
            return y
