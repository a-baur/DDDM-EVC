# Adapted from https://github.com/openai/jukebox

import math

import torch
import torch.nn as nn

import modules.vqvae.dist as dist


class ResConvBlock(nn.Module):
    def __init__(self, n_in: int, n_state: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(n_in, n_state, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_state, n_in, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


class Resnet(nn.Module):
    def __init__(self, n_in: int, n_depth: int, m_conv: float = 1.0) -> None:
        super().__init__()
        self.model = nn.Sequential(
            *[ResConvBlock(n_in, int(m_conv * n_in)) for _ in range(n_depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResConv1DBlock(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_state: int,
        dilation: int = 1,
        zero_out: bool = False,
        res_scale: float = 1.0,
    ) -> None:
        super().__init__()
        padding = dilation
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_in, n_state, 3, 1, padding, dilation),
            nn.ReLU(),
            nn.Conv1d(n_state, n_in, 1, 1, 0),
        )
        if zero_out:
            out = self.model[-1]
            nn.init.zeros_(out.weight)
            nn.init.zeros_(out.bias)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * self.model(x)


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_depth: int,
        m_conv: float = 1.0,
        dilation_growth_rate: int = 1,
        dilation_cycle: int = None,
        zero_out: bool = False,
        res_scale: bool = False,
        reverse_dilation: bool = False,
        checkpoint_res: bool = False,
    ) -> None:
        super().__init__()

        def _get_depth(depth: int) -> int:
            if dilation_cycle is None:
                return depth
            else:
                return depth % dilation_cycle

        blocks = [
            ResConv1DBlock(
                n_in,
                int(m_conv * n_in),
                dilation=dilation_growth_rate ** _get_depth(depth),
                zero_out=zero_out,
                res_scale=1.0 if not res_scale else 1.0 / math.sqrt(n_depth),
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]
        self.checkpoint_res = checkpoint_res
        if self.checkpoint_res == 1:
            if dist.get_rank() == 0:
                print("Checkpointing convs")
            self.blocks = nn.ModuleList(blocks)
        else:
            self.model = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint_res == 1:
            raise NotImplementedError("Checkpoint not implemented")
        else:
            return self.model(x)
