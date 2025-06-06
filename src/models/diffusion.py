import math
from typing import Literal

import torch

from config import DiffusionConfig
from modules.diffusion import GradLogPEstimatorV1, GradLogPEstimatorV2


class Diffusion(torch.nn.Module):
    """
    Decoupled Diffusion model.

    Use the sum of two decoupled score estimators
    to update the input tensor in a diffusion process.

    :param cfg: DiffusionConfig object.
    """

    def __init__(self, cfg: DiffusionConfig, score_model_ver: int = 1) -> None:
        super(Diffusion, self).__init__()

        self.n_feats = cfg.in_dim
        self.dim_unet = cfg.dec_dim
        self.dim_cond = cfg.cond_dim
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max

        if score_model_ver == 1:
            self.estimator_src = GradLogPEstimatorV1(
                cfg.dec_dim,
                cfg.cond_dim,
                cfg.gin_channels,
            )
            self.estimator_ftr = GradLogPEstimatorV1(
                cfg.dec_dim,
                cfg.cond_dim,
                cfg.gin_channels,
            )
        elif score_model_ver == 2:
            self.estimator_src = GradLogPEstimatorV2(
                cfg.in_dim,
                cfg.cond_dim,
                cfg.gin_channels,
                use_prior_conditioning=True,
            )
            self.estimator_ftr = GradLogPEstimatorV2(
                cfg.in_dim,
                cfg.cond_dim,
                cfg.gin_channels,
                use_prior_conditioning=True,
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

    def compute_diffused_mean(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        t: float | torch.Tensor,
        use_torch: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the diffused mean.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param t: Time step.
        :param use_torch: Whether to use torch or not.
        :return: Diffused mean.
        """
        x0_weight = self.get_gamma(0, t, use_torch=use_torch)
        mean_weight = 1.0 - x0_weight
        xt_src = x0 * x0_weight + src_out * mean_weight
        xt_ftr = x0 * x0_weight + ftr_out * mean_weight
        return xt_src * mask, xt_ftr * mask

    def forward_diffusion(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        t: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param t: Time step.
        :return: Diffused source, filter, and noise tensors.
        """
        xt_src, xt_ftr = self.compute_diffused_mean(
            x0, mask, src_out, ftr_out, t, use_torch=True
        )
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt_src = xt_src + z * torch.sqrt(variance)
        xt_ftr = xt_ftr + z * torch.sqrt(variance)

        return xt_src * mask, xt_ftr * mask, z * mask

    def reverse_ode(
        self,
        z_src: torch.Tensor,
        z_ftr: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        g: torch.Tensor,
        n_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse process using the probability flow ODE.

        The reverse process is defined as:
        dx_t = -0.5 * beta_t * (x_t - x_0) dt + 0.5 * beta_t * h * score(x_t, t) dt

        :param z_src: Latent source tensor.
        :param z_ftr: Latent filter tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param g: Global conditioning tensor.
        :param n_timesteps: Number of diffusion steps.
        :param mode: Inference mode (pf, em, ml).
        :return: Deterministic reconstruction of source and filter.
        """
        h = 1.0 / n_timesteps
        xt_src = z_src * mask
        xt_ftr = z_ftr * mask

        for i in range(n_timesteps):
            t = 1.0 - i * h
            time = t * torch.ones(
                z_src.shape[0], dtype=z_src.dtype, device=z_src.device
            )

            beta_t = self.get_beta(t)
            dxt_src = (src_out - xt_src) * (0.5 * beta_t * h)
            dxt_ftr = (ftr_out - xt_ftr) * (0.5 * beta_t * h)

            estimated_score = (
                0.5
                * (
                    self.estimator_src(xt_src, mask, src_out, g, time)
                    + self.estimator_ftr(xt_ftr, mask, ftr_out, g, time)
                )
                * (beta_t * h)
            )
            dxt_src -= estimated_score
            dxt_ftr -= estimated_score

            xt_src = (xt_src - dxt_src) * mask
            xt_ftr = (xt_ftr - dxt_ftr) * mask

        return xt_src, xt_ftr

    @torch.no_grad()  # type: ignore
    def reverse_diffusion(
        self,
        z_src: torch.Tensor,
        z_ftr: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        g: torch.Tensor,
        n_timesteps: int,
        mode: Literal["em", "ml"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reverse diffusion step.

        If mode is 'ml', the fast maximum likelihood sampling scheme
        by Popov et al. is used.
        See Theorem 1: https://arxiv.org/abs/2109.13821

        :param z_src: Latent source tensor.
        :param z_ftr: Latent filter tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param g: Global conditioning tensor.
        :param n_timesteps: Number of diffusion steps.
        :param mode: Inference mode (em, ml).
        :return: Updated source and filter tensors.
        """
        h = 1.0 / n_timesteps
        xt_src = z_src * mask
        xt_ftr = z_ftr * mask
        for i in range(n_timesteps):
            t = 1.0 - i * h
            time = t * torch.ones(
                z_src.shape[0], dtype=z_src.dtype, device=z_src.device
            )
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

            dxt_src = (src_out - xt_src) * (0.5 * beta_t * h + omega)
            dxt_ftr = (ftr_out - xt_ftr) * (0.5 * beta_t * h + omega)

            estimated_score = (
                (
                    self.estimator_src(xt_src, mask, src_out, g, time)
                    + self.estimator_ftr(xt_ftr, mask, ftr_out, g, time)
                )
                * (1.0 + kappa)
                * (beta_t * h)
            )
            dxt_src -= estimated_score
            dxt_ftr -= estimated_score

            sigma_n = torch.randn_like(z_src, device=z_src.device) * sigma
            dxt_src += sigma_n
            dxt_ftr += sigma_n

            xt_src = (xt_src - dxt_src) * mask
            xt_ftr = (xt_ftr - dxt_ftr) * mask

        return xt_src, xt_ftr

    def forward(
        self,
        z_src: torch.Tensor,
        z_ftr: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        g: torch.Tensor,
        n_timesteps: int,
        mode: Literal["pf", "em", "ml"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion model.

        :param z_src: Latent source tensor.
        :param z_ftr: Latent filter tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param g: Global conditioning tensor.
        :param n_timesteps: Number of diffusion steps.
        :param mode: Inference mode (pf, em, ml).
        :return: Updated source and filter tensors.
        """
        if mode in ["ml", "em"]:
            return self.reverse_diffusion(
                z_src, z_ftr, mask, src_out, ftr_out, g, n_timesteps, mode
            )
        elif mode == "pf":
            return self.reverse_ode(
                z_src, z_ftr, mask, src_out, ftr_out, g, n_timesteps
            )
        else:
            raise ValueError(
                f"Invalid mode: {mode}. Expected one of ['pf', 'em', 'ml']"
            )

    def loss_t(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        g: torch.Tensor,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for time step t.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param g: Global conditioning tensor.
        :param t: Time step.
        :return: Loss value.
        """
        xt_src, xt_ftr, z = self.forward_diffusion(x0, mask, src_out, ftr_out, t)

        z_estimation = self.estimator_src(xt_src, mask, src_out, g, t)
        z_estimation += self.estimator_ftr(xt_ftr, mask, ftr_out, g, t)

        z_estimation *= torch.sqrt(1.0 - self.get_gamma(0, t, p=2.0, use_torch=True))
        loss = torch.sum((z_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)

        return loss

    def compute_loss(
        self,
        x0: torch.Tensor,
        mask: torch.Tensor,
        src_out: torch.Tensor,
        ftr_out: torch.Tensor,
        g: torch.Tensor,
        offset: float = 1e-5,
    ) -> torch.Tensor:
        """
        Compute the loss for the diffusion model.

        :param x0: Initial input tensor.
        :param mask: Mask for the input tensor.
        :param src_out: Source output of Source-Filter encoder.
        :param ftr_out: Filter output of Source-Filter encoder.
        :param g: Global conditioning tensor.
        :param offset: Offset value to avoid numerical instability.
        :return: Loss value.
        """
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.loss_t(x0, mask, src_out, ftr_out, g, t)
