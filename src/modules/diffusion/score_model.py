import torch

from .modules import (
    Block,
    Downsample,
    LinearAttention,
    Mish,
    Residual,
    ResnetBlock,
    Rezero,
    SinusoidalPosEmb,
    Upsample,
)


class GradLogPEstimator(torch.nn.Module):
    """
    Score model for the diffusion model.

    :param dim_base: Base dimension.
    :param dim_cond: Condition dimension.
    :param gin_channels: Dimension of global conditioning tensor.
    :param dim_mults: Multipliers for downsampling and upsampling.
    """

    def __init__(
        self,
        dim_base: int,
        dim_cond: int,
        gin_channels: int,
        dim_mults: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super(GradLogPEstimator, self).__init__()

        dims = [2 + dim_cond, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_base, dim_base * 4),
            Mish(),
            torch.nn.Linear(dim_base * 4, dim_base),
        )
        cond_total = dim_base + gin_channels
        self.cond_block = torch.nn.Sequential(
            torch.nn.Linear(cond_total, 4 * dim_cond),
            Mish(),
            torch.nn.Linear(4 * dim_cond, dim_cond),
        )

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        enc_out: torch.Tensor,
        g: torch.Tensor,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass, estimating the gradient of the log-likelihood.

        :param x: Input tensor.
        :param x_mask: Mask for the input tensor.
        :param enc_out:
        :param g:
        :param t:
        :return:
        """
        condition = self.time_pos_emb(t)
        t = self.mlp(condition)

        x = torch.stack([enc_out, x], 1)
        x_mask = x_mask.unsqueeze(1)

        condition = torch.cat([condition, g.squeeze(-1)], 1)
        condition = self.cond_block(condition).unsqueeze(-1).unsqueeze(-1)

        condition = condition.expand(-1, -1, x.shape[2], x.shape[3]).contiguous()
        x = torch.cat([x, condition], 1)

        hiddens = []
        masks = [x_mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down.narrow(3, 0, mask_down.shape[3] // 2))

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)
        return (output * x_mask).squeeze(1)


class TokenScoreEstimator(torch.nn.Module):
    """
    Score model for the diffusion model.

    :param dim_base: Base dimension.
    :param
    :param gin_channels: Dimension of global conditioning tensor.
    :param dim_mults: Multipliers for downsampling and upsampling.
    """

    def __init__(
        self,
        n_feats: int,
        dim_base: int,
        gin_channels: int,
        dim_mults: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super(TokenScoreEstimator, self).__init__()

        dims = [2, *map(lambda m: dim_base * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_base, dim_base * 4),
            Mish(),
            torch.nn.Linear(dim_base * 4, dim_base),
        )

        self.cond_block = torch.nn.Sequential(
            torch.nn.Conv1d(gin_channels, 4 * gin_channels, 1),
            Mish(),
            torch.nn.Conv1d(4 * gin_channels, n_feats, 1),
        )

        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])

        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim_base),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim_base),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim_base)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim_base),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim_base),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )
        self.final_block = Block(dim_base, dim_base)
        self.final_conv = torch.nn.Conv2d(dim_base, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        t: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass, estimating the gradient of the log-likelihood.

        :param x: Input tensor.
        :param x_mask: Mask for the input tensor.
        :param enc_out:
        :param g:
        :param t:
        :return:
        """
        t = self.time_pos_emb(t)
        t = self.mlp(t)

        condition = self.cond_block(g)
        x_mask = x_mask.unsqueeze(1)
        x = torch.stack([x, condition], 1)

        hiddens = []
        masks = [x_mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down.narrow(3, 0, mask_down.shape[3] // 2))

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, x_mask)
        output = self.final_conv(x * x_mask)
        return (output * x_mask).squeeze(1)
