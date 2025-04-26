import math
from typing import Literal

import torch
from torch.nn.functional import l1_loss

from config import DiffusionConfig
from modules.diffusion.score_model import TokenScoreEstimator


class TokenDiffusion(torch.nn.Module):
    """
    Decoupled Diffusion model.

    Use the sum of two decoupled score estimators
    to update the input tensor in a diffusion process.

    :param cfg: DiffusionConfig object.
    """

    def __init__(self, cfg: DiffusionConfig) -> None:
        super(TokenDiffusion, self).__init__()

        self.n_feats = cfg.in_dim
        self.dim_unet = cfg.dec_dim
        self.dim_cond = cfg.cond_dim
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max

        self.estimator_src = TokenScoreEstimator(
            cfg.dec_dim, cfg.cond_dim, cfg.gin_channels
        )
        self.estimator_ftr = TokenScoreEstimator(
            cfg.dec_dim, cfg.cond_dim, cfg.gin_channels
        )

    def get_beta(self, t: float | torch.Tensor) -> float | torch.Tensor:
        """
        Compute beta function.

        :param t: time step.
        :return: beta value.
        """
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(
        self,
        s: float | torch.Tensor,
        t: float | torch.Tensor,
        p: float | torch.Tensor = 1.0,
        use_torch: bool = False,
    ) -> float | torch.Tensor:
        """
        Compute gamma function.

        :param s: start time.
        :param t: end time.
        :param p: scaling factor.
        :param use_torch: whether to use torch or not.
        :return: gamma value.
        """
        beta_integral = self.beta_min + 0.5 * (self.beta_max - self.beta_min) * (t + s)
        beta_integral *= t - s
        if use_torch:
            gamma = torch.exp(-0.5 * p * beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5 * p * beta_integral)
        return gamma

    def get_mu(
        self, s: float | torch.Tensor, t: float | torch.Tensor
    ) -> float | torch.Tensor:
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(
        self, s: float | torch.Tensor, t: float | torch.Tensor
    ) -> float | torch.Tensor:
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(
        self, s: float | torch.Tensor, t: float | torch.Tensor
    ) -> float | torch.Tensor:
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        t: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param t: Time step.
        :return: Diffused source, filter, and noise tensors.
        """
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = x0 + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()  # type: ignore
    def reverse_diffusion(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        src_tkn: torch.Tensor,
        ftr_tkn: torch.Tensor,
        n_timesteps: int,
        mode: Literal["em", "ml"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse diffusion step.

        If mode is 'ml', the fast maximum likelihood sampling scheme
        by Popov et al. is used.
        See Theorem 1: https://arxiv.org/abs/2109.13821

        :param z: Latent noise tensor.
        :param mask: Mask for the input tensor.
        :param g: Global conditioning tensor.
        :param n_timesteps: Number of diffusion steps.
        :param mode: Inference mode (em, ml).
        :return: Updated source and filter tensors.
        """
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i * h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)

            if mode == "ml":
                # Theorem 1: https://arxiv.org/abs/2109.13821
                kappa = self.get_gamma(0, t - h) * (
                    1.0 - self.get_gamma(t - h, t, p=2.0)
                )
                kappa /= self.get_gamma(0, t) * beta_t * h
                kappa -= 1.0
                omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                omega += self.get_mu(t - h, t)
                omega -= 0.5 * beta_t * h + 1.0
                sigma = self.get_sigma(t - h, t)
            else:
                kappa = 0.0
                omega = 0.0
                sigma = math.sqrt(beta_t * h)

            dxt = xt * (0.5 * beta_t * h + omega)

            estimated_score = (
                (
                    self.estimator_src(xt, mask, src_tkn, time)
                    + self.estimator_ftr(xt, mask, ftr_tkn, time)
                )
                * (1.0 + kappa)
                * (beta_t * h)
            )
            dxt -= estimated_score

            sigma_n = torch.randn_like(z, device=z.device) * sigma
            dxt += sigma_n

            xt = (xt - dxt) * mask

        return xt

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        src_tkn: torch.Tensor,
        ftr_tkn: torch.Tensor,
        n_timesteps: int,
        mode: Literal["em", "ml"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion model.

        :param z: Latent noise tensor.
        :param mask: Mask for the input tensor.
        :param g: Global conditioning tensor.
        :param n_timesteps: Number of diffusion steps.
        :param mode: Inference mode (pf, em, ml).
        :return: Updated source and filter tensors.
        """
        return self.reverse_diffusion(z, mask, src_tkn, ftr_tkn, n_timesteps, mode)

    def loss_t(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        src_tkn: torch.Tensor,
        ftr_tkn: torch.Tensor,
        t: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for time step t.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param t: Time step.
        :return: Loss value.
        """
        xt, z = self.forward_diffusion(x0, mask, t)

        z_estimation = self.estimator_src(xt, mask, src_tkn, t)
        z_estimation += self.estimator_ftr(xt, mask, ftr_tkn, t)
        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))

        alpha_t: torch.Tensor = self.get_gamma(0, t, p=2.0, use_torch=True)

        use_snr_weighting = False
        if use_snr_weighting:
            snr_weight = alpha_t / (1.0 - alpha_t + 1e-5)
            snr_weight = torch.clamp(snr_weight, max=3.0)
            snr_weight = snr_weight.detach()
            score_loss = torch.sum(snr_weight * (z_estimation + z) ** 2) / (
                torch.sum(mask) * self.n_feats
            )
        else:
            score_loss = torch.sum((z_estimation + z) ** 2) / (
                torch.sum(mask) * self.n_feats
            )

        x_hat = (xt - torch.sqrt(1 - alpha_t) * z_estimation) / torch.sqrt(alpha_t)
        rec_loss = l1_loss(x_hat, x0, reduction="mean")

        return score_loss, rec_loss

    def compute_loss(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        src_tkn: torch.Tensor,
        ftr_tkn: torch.Tensor,
        offset: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss for the diffusion model.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param g: Global conditioning tensor.
        :param offset: Offset value to avoid numerical instability.
        :return: Loss value.
        """
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.loss_t(x0, mask, src_tkn, ftr_tkn, t)
