import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from models.dddm.preprocessor import DDDMInput
from models.diffusion import Diffusion
from models.token_diffusion import TokenDiffusion
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
        rec_loss: bool = False,
        time_steps: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the resynthesis loss of the DDDM model.

        :param x: DDDM input object
        :param g: Global conditioning tensor
        :param rec_loss: Whether to compute the diffusion reconstruction loss or not
        :param time_steps: Number of diffusion steps for reconstruction loss
        :return: Tuple of (score loss, source-filter loss, reconstruction loss)
        """
        mixup_ratios = torch.randint(0, 2, (x.batch_size,)).to(x.device).detach()
        src_mel, ftr_mel, src_mixup, ftr_mixup = self.encoder(x, g, mixup_ratios)
        src_ftr_loss = F.l1_loss(x.mel, src_mel + ftr_mel)

        if rec_loss:
            src_mean_x, ftr_mean_x = self.diffusion.compute_diffused_mean(
                x.mel, x.mask, src_mixup, ftr_mixup, 1.0
            )

        # compute the diffusion loss on mixed up outputs
        max_length_new = util.get_u_net_compatible_length(x.mel.size(-1))
        src_mixup, ftr_mixup, x_pad, x_mask_pad = util.pad_tensors_to_length(
            [src_mixup, ftr_mixup, x.mel, x.mask], max_length_new
        )
        diff_loss = self.diffusion.compute_loss(
            x_pad, x_mask_pad, src_mixup, ftr_mixup, g
        )

        if rec_loss:
            # Pad the sequences for U-Net compatibility
            src_mean_x, ftr_mean_x = util.pad_tensors_to_length(
                [src_mean_x, ftr_mean_x], max_length_new
            )

            # Add noise to diffused mean to create priors for diffusion
            start_n = torch.randn_like(src_mean_x, device=src_mean_x.device)
            src_mean_x.add_(start_n)
            ftr_mean_x.add_(start_n)

            # use probability flow ODE to maintain gradient flow
            y_mel = self.diffusion(
                src_mean_x,
                ftr_mean_x,
                x_mask_pad,
                src_mixup,
                ftr_mixup,
                g,
                time_steps,
                "pf",
            )
            rec_loss = F.l1_loss(y_mel, x_pad)
            return diff_loss, src_ftr_loss, rec_loss
        else:
            return diff_loss, src_ftr_loss, torch.tensor(0.0)

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
            _src_mel, _ftr_mel = src_mel.clone().detach(), ftr_mel.clone().detach()
        else:
            _src_mel, _ftr_mel = None, None

        # Compute diffused mean
        src_mean_x, ftr_mean_x = self.diffusion.compute_diffused_mean(
            x.mel, x.mask, src_mel, ftr_mel, 1.0
        )

        # Pad the sequences for U-Net compatibility
        max_length_new = util.get_u_net_compatible_length(x.mel.size(-1))
        src_mean_x, ftr_mean_x, src_mel, ftr_mel, x_mask = util.pad_tensors_to_length(
            [src_mean_x, ftr_mean_x, src_mel, ftr_mel, x.mask], max_length_new
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
        y = y[:, :, : x.mel.size(-1)]  # Remove the padded frames

        if return_enc_out:
            return y, _src_mel, _ftr_mel
        else:
            return y


class TokenDDDM(nn.Module):
    """
    Decoupled Denoising Diffusion model for emotional voice conversion

    Instead of data-driven priors, this model uses Gaussian priors
    and is conditioned on pitch and filter tokens.

    :param encoder: Source-Filter encoder
    :param diffusion: Diffusion model
    """

    def __init__(self, token_encoder: nn.Module, diffusion: TokenDiffusion) -> None:
        super().__init__()

        self.token_encoder = token_encoder
        self.diffusion = diffusion

    def compute_loss(
        self, x: DDDMInput, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the resynthesis loss of the DDDM model.

        :param x: DDDM input object
        :param g: Global conditioning tensor
        :return: Tuple of (score loss, source-filter loss, reconstruction loss)
        """
        src_tkn, ftr_tkn = self.token_encoder(x, g)

        # compute the diffusion loss on mixed up outputs
        max_length_new = util.get_u_net_compatible_length(x.mel.size(-1))
        src_tkn, ftr_tkn, x_pad, x_mask_pad = util.pad_tensors_to_length(
            [src_tkn, ftr_tkn, x.mel, x.mask], max_length_new
        )
        diff_loss, rec_loss = self.diffusion.compute_loss(
            x_pad, x_mask_pad, src_tkn, ftr_tkn
        )

        return diff_loss, rec_loss

    def forward(
        self,
        x: DDDMInput,
        g: torch.Tensor,
        n_time_steps: int = 6,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DDDM model.

        :param x: DDDM input object
        :param g: Global conditioning tensor
        :param n_time_steps: Number of diffusion steps
        :return: Output mel-spectrogram (and encoder output if return_enc_out is True)
        """
        src_tkn, ftr_tkn = self.token_encoder(x, g)

        # Add noise to diffused mean to create priors for diffusion
        z = torch.randn_like(x.mel, device=src_tkn.device)

        # Pad the sequences for U-Net compatibility
        max_length_new = util.get_u_net_compatible_length(x.mel.size(-1))
        z, src_tkn, ftr_tkn, x_mask = util.pad_tensors_to_length(
            [z, src_tkn, ftr_tkn, x.mask], max_length_new
        )

        # Diffusion
        y = self.diffusion(z, x_mask, src_tkn, ftr_tkn, n_time_steps)
        y = y[:, :, : x.mel.size(-1)]  # Remove the padded frames

        return y
