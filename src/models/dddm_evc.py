import torch
import torch.nn as nn

import util
from config import ModelsConfig

from .content_encoder import Wav2Vec2
from .diffusion import Diffusion
from .pitch_encoder import VQVAE
from .speaker_encoder import MetaStyleSpeech
from .wavenet_decoder import WavenetDecoder


class DDDM(nn.Module):
    def __init__(self, cfg: ModelsConfig) -> None:
        super().__init__()
        self.content_encoder = Wav2Vec2()
        self.pitch_encoder = VQVAE(cfg.pitch_encoder)
        self.speaker_encoder = MetaStyleSpeech(cfg.speaker_encoder)
        self.decoder = WavenetDecoder(cfg)
        self.diffusion = Diffusion(cfg.diffusion)

    def forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_mask: torch.Tensor,
        mixup: bool = False,
        sample_rate: int = 16000,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_mask: Mask for the input mel-spectrogram
        :param mixup: Whether to use prior mixup or not
        :param sample_rate: Sampling rate of the input waveform
        :param return_enc_out: Whether to return the encoder output or not
        :return: Source, and Filter representations
        """
        f0 = util.get_normalized_f0(x, sample_rate)

        x_emb_content = self.content_encoder(x)
        x_emb_pitch = self.pitch_encoder.code_extraction(f0)
        x_emb_spk = self.speaker_encoder(x_mel, x_mask).unsqueeze(-1)

        src_mel, ftr_mel = self.decoder(
            x_emb_spk, x_emb_content, x_emb_pitch, x_mask, mixup=mixup
        )

        # Pad the sequences for U-Net compatibility
        # Move to diffusion model?
        # -> self.diffusion.prepare_input(x_mel, x_mask, src_mel, ftr_mel)
        # -> returns src_z, ftr_z, src_mel, ftr_mel, mel_mask
        x_mel_lengths = torch.LongTensor([x_mel.size(-1)])
        x_mel_mask = util.sequence_mask(x_mel_lengths, x_mel.size(2)).to(x_mel.dtype)

        src_mean_x, ftr_mean_x = self.diffusion.compute_diffused_mean(
            x_mel, x_mask, src_mel, ftr_mel, 1.0
        )

        max_length = int(x_mel_lengths.max())
        max_length_new = util.get_u_net_compatible_length(max_length)

        src_mean_x = util.pad_to_length(src_mean_x, max_length_new)
        ftr_mean_x = util.pad_to_length(ftr_mean_x, max_length_new)
        src_mel = util.pad_to_length(src_mel, max_length_new)
        ftr_mel = util.pad_to_length(ftr_mel, max_length_new)

        if max_length_new > max_length:
            x_mel_mask = util.sequence_mask(x_mel_lengths, max_length_new).to(
                x_mel.dtype
            )

        # Add noise to diffused mean to create priors for diffusion
        start_n = torch.randn_like(src_mean_x, device=src_mean_x.device)
        src_mean_x.add_(start_n)
        ftr_mean_x.add_(start_n)

        # Diffusion
        y_src, y_ftr = self.diffusion(
            src_mean_x, ftr_mean_x, x_mel_mask, src_mel, ftr_mel, x_emb_spk, 6, "ml"
        )
        y = (y_src + y_ftr) / 2

        if return_enc_out:
            enc_out = src_mel + ftr_mel
            return y, enc_out
        else:
            return y
