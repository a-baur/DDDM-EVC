from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from config import ConfigEVC, ConfigVC
from models.diffusion import Diffusion
from models.source_filter_encoder import SourceFilterEncoder
from models.style_encoder import MetaStyleSpeech, StyleEncoder


class Pretrained(Enum):
    METASTYLE_SPEECH = "metastylespeech.pth"
    VQVAE = "vqvae.pth"
    VC_WAVENET = "vc/wavenet_decoder.pth"
    VC_DIFFUSION = "vc/diffusion.pth"
    EVC_STYLEENCODER = "evc/style_encoder.pth"
    EVC_WAVENET = "evc/wavenet_decoder.pth"
    EVC_DIFFUSION = "evc/diffusion.pth"


class DDDM(nn.Module):
    """Decoupled Denoising Diffusion model for emotional voice conversion"""

    def __init__(
        self,
        style_encoder: StyleEncoder | MetaStyleSpeech,
        src_ftr_encoder: SourceFilterEncoder,
        diffusion: Diffusion,
    ) -> None:
        super().__init__()

        self.style_encoder = style_encoder
        self.encoder = src_ftr_encoder
        self.diffusion = diffusion

    def voice_conversion(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_n_frames: torch.Tensor,
        y: torch.Tensor,
        y_mel: torch.Tensor,
        y_n_frames: torch.Tensor,
        n_time_steps: int = 6,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Voice conversion using the Source-Filter encoder.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_n_frames: Number of unpaded frames in the input mel-spectrogram
        :param y: Target waveform
        :param y_mel: Target mel-spectrogram
        :param y_n_frames: Number of unpaded frames in the target mel-spectrogram
        :param n_time_steps: Number of diffusion steps
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)

        # Get the global conditioning tensor for the target speaker
        y_mask = util.sequence_mask(y_n_frames, y_mel.size(2)).to(y_mel.dtype)
        g = self.style_encoder(y, y_mel, y_mask).unsqueeze(-1)  # (B, C, 1)

        return self._forward(x, x_mel, x_mask, g, n_time_steps, return_enc_out)

    def forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_n_frames: torch.Tensor,
        n_time_steps: int = 6,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DDDM model.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_n_frames: Number of unpaded frames in the input mel-spectrogram
        :param n_time_steps: Number of diffusion steps
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        # Encode the input waveform into diffusion priors
        x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)
        g = self.style_encoder(x, x_mel, x_mask).unsqueeze(-1)  # (B, C, 1)
        return self._forward(x, x_mel, x_mask, g, n_time_steps, return_enc_out)

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
        g = self.style_encoder(x, x_mel, x_mask).unsqueeze(-1)  # (B, C, 1)

        mixup_ratios = torch.randint(0, 2, (x.size(0),)).to(x.device)
        src_mel, ftr_mel = self.encoder(x, x_mask, g, mixup_ratios)

        max_length_new = util.get_u_net_compatible_length(x_mel.size(-1))
        src_mel, ftr_mel, x_mel, x_mask = util.pad_tensors_to_length(
            [src_mel, ftr_mel, x_mel, x_mask], max_length_new
        )

        diff_loss = self.diffusion.compute_loss(x_mel, x_mask, src_mel, ftr_mel, g)
        rec_loss = F.l1_loss(x_mel, src_mel + ftr_mel)
        return diff_loss, rec_loss

    def _forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        n_time_steps: int,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with the given global conditioning tensor.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_mask: Mask for the input mel-spectrogram
        :param g: Global conditioning tensor
        :param n_time_steps: Number of diffusion steps
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram or (y_mel, src_mel, ftr_mel) if return_enc_out
        """
        # Encode the input waveform into diffusion priors
        src_mel, ftr_mel = self.encoder(x, x_mask, g)

        if return_enc_out:
            _src_mel, _ftr_mel = src_mel.detach().clone(), ftr_mel.detach().clone()
        else:
            _src_mel, _ftr_mel = None, None

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
            src_mean_x, ftr_mean_x, x_mask, src_mel, ftr_mel, g, n_time_steps, "ml"
        )
        y = (y_src + y_ftr) / 2
        y = y[:, :, : x_mel.size(-1)]  # Remove the padded frames

        if return_enc_out:
            return y, _src_mel, _ftr_mel
        else:
            return y

    @classmethod
    def from_config(cls, cfg: ConfigVC | ConfigEVC, pretrained: bool = False) -> "DDDM":
        """
        Initialize DDDM model from configuration file.

        Always uses pretrained speaker and emotion model.

        :param cfg: ConfigVC or ConfigEVC
        :param pretrained: If true, load pretrained models
        :return: DDDM model
        """
        src_ftr_encoder = SourceFilterEncoder(cfg.model, cfg.data.dataset.sampling_rate)
        util.load_model(src_ftr_encoder.pitch_encoder, Pretrained.VQVAE.value)
        diffusion = Diffusion(cfg.model.diffusion)

        if isinstance(cfg, ConfigVC):
            style_encoder = MetaStyleSpeech(cfg.model.speaker_encoder)
            util.load_model(style_encoder, Pretrained.METASTYLE_SPEECH.value)

            style_encoder.requires_grad_(False)

            if pretrained:
                util.load_model(src_ftr_encoder.decoder, Pretrained.VC_WAVENET.value)
                util.load_model(diffusion, Pretrained.VC_DIFFUSION.value)

        elif isinstance(cfg, ConfigEVC):
            style_encoder = StyleEncoder(cfg.model.style_encoder)
            util.load_model(
                style_encoder.speaker_encoder, Pretrained.METASTYLE_SPEECH.value
            )

            style_encoder.emotion_encoder.requires_grad_(False)
            style_encoder.speaker_encoder.requires_grad_(False)

            if pretrained:
                util.load_model(src_ftr_encoder.decoder, Pretrained.EVC_WAVENET.value)
                util.load_model(diffusion, Pretrained.EVC_DIFFUSION.value)
                util.load_model(style_encoder, Pretrained.EVC_STYLEENCODER.value)

        else:
            raise ValueError(f"Unknown config type: {type(cfg).__name__}")

        if pretrained:
            src_ftr_encoder.requires_grad_(False)
            diffusion.requires_grad_(False)
            style_encoder.requires_grad_(False)

        return cls(style_encoder, src_ftr_encoder, diffusion)
