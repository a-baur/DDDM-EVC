import torch
from torch.nn.functional import l1_loss

from config import DiffusionConfig
from modules.diffusion.score_model import TokenScoreEstimator

PI = torch.pi


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
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max

        self.use_snr_weighting = True

        self.s = 0.008

        self.estimator_src = TokenScoreEstimator(
            cfg.in_dim, cfg.dec_dim, cfg.gin_channels
        )
        self.estimator_ftr = TokenScoreEstimator(
            cfg.in_dim, cfg.dec_dim, cfg.gin_channels
        )

    def get_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative alpha_bar(t) following cosine schedule.
        t: Tensor of shape (any).
        returns: Tensor of shape (same as t).
        """
        if isinstance(t, float):
            t = torch.tensor(t, dtype=torch.float32)

        inner = (t + self.s) / (1.0 + self.s)
        alpha_t = torch.cos(inner * PI / 2) ** 2
        return alpha_t

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
        alphabar = self.get_alpha_bar(t).view(-1, 1, 1)  # ᾱ(t)
        noise = torch.randn_like(x0)
        xt = (alphabar.sqrt() * x0 + (1.0 - alphabar).sqrt() * noise) * mask
        return xt, noise * mask

    @torch.no_grad()  # type: ignore
    def reverse_diffusion(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        src_tkn: torch.Tensor,
        ftr_tkn: torch.Tensor,
        n_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-compute cosine schedule
        t_grid = torch.linspace(0, 1, n_timesteps + 1, device=z.device)
        alphabars = self.get_alpha_bar(t_grid).to(z.device)
        alphabars[-1] = 1e-5
        alphas = alphabars[1:] / alphabars[:-1]
        betas = 1.0 - alphas
        sigmas = torch.sqrt(betas * (1.0 - alphabars[:-1]) / (1.0 - alphabars[1:]))

        # Perform reverse diffusion
        xt = z * mask
        for t in range(n_timesteps, 0, -1):  # t = N … 1
            # continuous time fed to the score nets
            t_cont = torch.full(
                (z.shape[0],), t / n_timesteps, dtype=z.dtype, device=z.device
            )

            eps_hat = self.estimator_src(xt, mask, src_tkn, t_cont)
            eps_hat += self.estimator_ftr(xt, mask, ftr_tkn, t_cont)

            alpha_t = alphas[t - 1]
            beta_t = betas[t - 1]
            abar_t = alphabars[t]
            sigma_t = sigmas[t - 1]

            mu = (xt - beta_t * eps_hat / torch.sqrt(1.0 - abar_t)) / torch.sqrt(
                alpha_t
            )

            if t > 1:
                xt = mu + sigma_t * torch.randn_like(xt)
            else:
                xt = mu
            xt *= mask

        return xt

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor,
        src_tkn: torch.Tensor,
        ftr_tkn: torch.Tensor,
        n_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion model.

        :param z: Latent noise tensor.
        :param mask: Mask for the input tensor.
        :param src_tkn: Source token tensor.
        :param ftr_tkn: Filter token tensor.
        :param n_timesteps: Number of diffusion steps.
        :return: Updated source and filter tensors.
        """
        return self.reverse_diffusion(z, mask, src_tkn, ftr_tkn, n_timesteps)

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
        :param src_tkn: Source token tensor.
        :param ftr_tkn: Filter token tensor.
        :param t: Time step.
        :return: Loss value.
        """
        xt, z = self.forward_diffusion(x0, mask, t)

        z_estimation = self.estimator_src(xt, mask, src_tkn, t)
        z_estimation += self.estimator_ftr(xt, mask, ftr_tkn, t)

        alpha_t = self.get_alpha_bar(t).unsqueeze(-1).unsqueeze(-1)

        if self.use_snr_weighting:
            gamma = 5.0  # Recommended value from the paper
            snr = alpha_t / (1.0 - alpha_t + 1e-5)
            snr_weight = torch.minimum(gamma / snr, torch.ones_like(snr))
            snr_weight = snr_weight.detach()

            score_loss = torch.sum(snr_weight * (z_estimation - z) ** 2) / (
                torch.sum(mask) * self.n_feats
            )
        else:
            score_loss = torch.sum((z_estimation - z) ** 2) / (
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
        :param src_tkn: Source token tensor.
        :param ftr_tkn: Filter token tensor.
        :param offset: Offset value to avoid numerical instability.
        :return: Loss value.
        """
        b = x0.shape[0]
        t = torch.rand(b, dtype=x0.dtype, device=x0.device, requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.loss_t(x0, mask, src_tkn, ftr_tkn, t)
