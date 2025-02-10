import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from config import ModelConfig

from .diffusion import Diffusion
from .source_filter_encoder import SourceFilterEncoder
from .speaker_encoder import MetaStyleSpeech


class DDDM(nn.Module):
    def __init__(self, cfg: ModelConfig, sample_rate: int) -> None:
        super().__init__()
        self.style_encoder = MetaStyleSpeech(cfg.speaker_encoder)
        self.source_filter_encoder = SourceFilterEncoder(cfg, sample_rate)
        self.diffusion = Diffusion(cfg.diffusion)

    def load_pretrained(self, freeze: bool = True) -> None:
        """Load pre-trained models."""
        util.load_model(self.style_encoder, "metastylespeech.pth", freeze=freeze)
        util.load_model(
            self.source_filter_encoder.pitch_encoder,
            ckpt_file="vqvae.pth",
            freeze=freeze,
        )
        util.load_model(
            self.source_filter_encoder.decoder,
            ckpt_file="wavenet_decoder.pth",
            freeze=freeze,
        )
        util.load_model(self.diffusion, "diffusion.pth", freeze=freeze)

    def voice_conversion(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_n_frames: torch.Tensor,
        y_mel: torch.Tensor,
        y_length: torch.Tensor,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Voice conversion using the Source-Filter encoder.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_n_frames: Number of unpaded frames in the input mel-spectrogram
        :param y_mel: Target mel-spectrogram
        :param y_length: Number of unpaded frames in the target mel-spectrogram
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)

        # Get the global conditioning tensor for the target speaker
        y_mask = util.sequence_mask(y_length, y_mel.size(2)).to(y_mel.dtype)
        g = self.style_encoder(y_mel, y_mask).unsqueeze(-1)  # (B, C, 1)

        return self._forward(x, x_mel, x_mask, g, return_enc_out)

    def forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_n_frames: torch.Tensor,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DDDM model.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_n_frames: Number of unpaded frames in the input mel-spectrogram
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        # Encode the input waveform into diffusion priors
        x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)
        g = self.style_encoder(x_mel, x_mask).unsqueeze(-1)  # (B, C, 1)
        return self._forward(x, x_mel, x_mask, g, return_enc_out)

    def _forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with the given global conditioning tensor.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_mask: Mask for the input mel-spectrogram
        :param g: Global conditioning tensor
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        # Encode the input waveform into diffusion priors
        src_mel, ftr_mel = self.source_filter_encoder(x, x_mask, g)
        enc_out = src_mel + ftr_mel

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
        y = y[:, :, : x_mel.size(-1)]  # Remove the padded frames

        if return_enc_out:
            return y, enc_out
        else:
            return y

    def compute_loss(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_n_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the resynthesis loss of the DDDM model.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_n_frames: Number of unpaded frames in the input mel-spectrogram
        :return: Tuple of the diffusion loss and the reconstruction loss
        """
        x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)
        g = self.style_encoder(x_mel, x_mask).unsqueeze(-1)  # (B, C, 1)

        mixup_ratios = torch.randint(0, 2, (x.size(0),)).to(x.device)
        src_mel, ftr_mel = self.source_filter_encoder(x, x_mask, g, mixup_ratios)

        max_length_new = util.get_u_net_compatible_length(x_mel.size(-1))
        src_mel, ftr_mel, x_mel, x_mask = util.pad_tensors_to_length(
            [src_mel, ftr_mel, x_mel, x_mask], max_length_new
        )

        diff_loss = self.diffusion.compute_loss(x_mel, x_mask, src_mel, ftr_mel, g)
        rec_loss = F.l1_loss(x_mel, src_mel + ftr_mel)
        return diff_loss, rec_loss
