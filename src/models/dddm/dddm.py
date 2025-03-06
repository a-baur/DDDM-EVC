import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from models.dddm.input import DDDMInput
from models.diffusion import Diffusion
from modules.wavenet_decoder import WavenetDecoder


class DDDM(nn.Module):
    """
    Decoupled Denoising Diffusion model for emotional voice conversion

    :param encoder: Source-Filter encoder
    :param diffusion: Diffusion model
    """

    def __init__(self, encoder: WavenetDecoder, diffusion: Diffusion) -> None:
        super().__init__()

        self.encoder = encoder
        self.diffusion = diffusion

    def compute_loss(
        self,
        x: DDDMInput,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the resynthesis loss of the DDDM model.

        :param x: DDDM input object
        :param g: Global conditioning tensor
        :return: Tuple of the diffusion loss and the reconstruction loss
        """
        mixup_ratios = torch.randint(0, 2, (x.batch_size,)).to(x.device).detach()
        src_mel, ftr_mel, src_mixup, ftr_mixup = self.encoder(x, g, mixup_ratios)

        max_length_new = util.get_u_net_compatible_length(x.mel.size(-1))
        src_mixup, ftr_mixup, x_pad, x_mask_pad = util.pad_tensors_to_length(
            [src_mixup, ftr_mixup, x.mel, x.mask], max_length_new
        )

        diff_loss = self.diffusion.compute_loss(
            x_pad, x_mask_pad, src_mixup, ftr_mixup, g
        )
        rec_loss = F.l1_loss(x.mel, src_mel + ftr_mel)
        return diff_loss, rec_loss

    def forward(
        self,
        x: DDDMInput,
        g: torch.Tensor,
        n_time_steps: int = 6,
        return_enc_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DDDM model.

        :param x: DDDM input object
        :param g: Global conditioning tensor
        :param n_time_steps: Number of diffusion steps
        :param return_enc_out: Whether to return the encoder output or not
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        src_mel, ftr_mel = self.encoder(x, g)

        if return_enc_out:
            _src_mel, _ftr_mel = src_mel.detach().clone(), ftr_mel.detach().clone()
        else:
            _src_mel, _ftr_mel = None, None

        # Compute diffused mean
        src_mean_x, ftr_mean_x = self.diffusion.compute_diffused_mean(
            x.mel, x.mask, src_mel, ftr_mel, 1.0
        )

        # Pad the sequences for U-Net compatibility
        max_length_new = util.get_u_net_compatible_length(x.mel.size(-1))
        src_mean_x, ftr_mean_x, src_mel, ftr_mel, x.mask = util.pad_tensors_to_length(
            [src_mean_x, ftr_mean_x, src_mel, ftr_mel, x.mask], max_length_new
        )

        # Add noise to diffused mean to create priors for diffusion
        start_n = torch.randn_like(src_mean_x, device=src_mean_x.device)
        src_mean_x.add_(start_n)
        ftr_mean_x.add_(start_n)

        # Diffusion
        y_src, y_ftr = self.diffusion(
            src_mean_x, ftr_mean_x, x.mask, src_mel, ftr_mel, g, n_time_steps, "ml"
        )
        y = (y_src + y_ftr) / 2
        y = y[:, :, : x.mel.size(-1)]  # Remove the padded frames

        if return_enc_out:
            return y, _src_mel, _ftr_mel
        else:
            return y
