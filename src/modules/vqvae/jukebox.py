# Adapted from https://github.com/openai/jukebox
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config import F0VAEConfig, Resnet1DConfig

from .resnet import Resnet1D


def assert_shape(x: torch.Tensor, exp_shape: tuple) -> None:
    """Assert the shape of a tensor is as expected."""
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"


def _get_filter_pad(s_t: int) -> tuple[int, int]:
    if s_t % 2 == 0:
        return s_t * 2, s_t // 2
    return s_t * 2 + 1, s_t // 2 + 1


class EncoderConvBlock(nn.Module):
    """
    Convolutional block for the encoder.
    """

    def __init__(
        self,
        input_emb_width: Any,
        output_emb_width: Any,
        down_t: int | tuple[int] | list[int],
        stride_t: int | tuple[int] | list[int],
        width: int,
        resnet1d: Resnet1DConfig,
    ) -> None:
        super().__init__()
        blocks = []

        if not isinstance(stride_t, (tuple, list)):
            stride_t, down_t = [stride_t], [down_t]

        for k, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
            if d_t == 0:
                continue

            filter_t, pad_t = _get_filter_pad(s_t)
            first_block = k == 0

            for i in range(d_t):
                first_dilation = i == 0
                block = nn.Sequential(
                    nn.Conv1d(
                        input_emb_width if first_dilation and first_block else width,
                        width,
                        filter_t,
                        s_t,
                        pad_t,
                    ),
                    Resnet1D(
                        width,
                        resnet1d.depth,
                        resnet1d.m_conv,
                        resnet1d.dilation_growth_rate,
                        resnet1d.dilation_cycle,
                        resnet1d.zero_out,
                        resnet1d.res_scale,
                    ),
                )
                blocks.append(block)
        block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
        blocks.append(block)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class DecoderConvBock(nn.Module):
    def __init__(
        self,
        input_emb_width: int,
        output_emb_width: int,
        down_t: int | tuple[int] | list[int],
        stride_t: int | tuple[int] | list[int],
        width: int,
        resnet1d: Resnet1DConfig,
    ):
        super().__init__()
        blocks = []

        if not isinstance(stride_t, (tuple, list)):
            stride_t, down_t = [stride_t], [down_t]

        block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
        blocks.append(block)

        for k, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
            if d_t == 0:
                continue

            filter_t, pad_t = _get_filter_pad(s_t)
            last_block = k == len(stride_t) - 1

            for i in range(d_t):
                last_dilation = i == (d_t - 1)
                block = nn.Sequential(
                    Resnet1D(
                        width,
                        resnet1d.depth,
                        resnet1d.m_conv,
                        resnet1d.dilation_growth_rate,
                        resnet1d.dilation_cycle,
                        resnet1d.zero_out,
                        resnet1d.res_scale,
                        resnet1d.reverse_dialation,
                        resnet1d.checkpoint_res,
                    ),
                    nn.ConvTranspose1d(
                        width,
                        input_emb_width if last_dilation and last_block else width,
                        filter_t,
                        s_t,
                        pad_t,
                    ),
                )
                blocks.append(block)

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, cfg: F0VAEConfig) -> None:
        super().__init__()
        self.input_emb_width = cfg.in_dim
        self.output_emb_width = cfg.out_dim
        self.levels = cfg.levels
        self.downs_t = cfg.downs_t
        self.strides_t = cfg.strides_t

        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            stride_t, down_t = self.strides_t[level], self.downs_t[level]
            block = EncoderConvBlock(
                cfg.in_dim if level == 0 else cfg.out_dim,
                cfg.out_dim,
                down_t=down_t,
                stride_t=stride_t,
                width=cfg.hidden_dim,
                resnet1d=cfg.resnet1d,
            )
            self.level_blocks.append(block)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass through the encoder.

        :param x: Input tensor
        :return: List of tensors at each level
        """
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            stride_t, down_t = self.strides_t[level], self.downs_t[level]
            x = level_block(x)

            # assert output has the correct shape
            # given the stride and downsample
            if isinstance(stride_t, (tuple, list)) and isinstance(
                down_t, (tuple, list)
            ):
                emb = self.output_emb_width
                T = T // np.prod([s**d for s, d in zip(stride_t, down_t)])
            else:
                emb, T = self.output_emb_width, T // (stride_t**down_t)
            assert_shape(x, (N, emb, T))

            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(self, cfg: F0VAEConfig):
        super().__init__()
        self.input_emb_width = cfg.in_dim
        self.output_emb_width = cfg.out_dim
        self.levels = cfg.levels

        self.strides_t, self.downs_t = cfg.strides_t, cfg.downs_t

        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            self.level_blocks.append(
                DecoderConvBock(
                    cfg.in_dim,
                    cfg.out_dim,
                    down_t=self.downs_t[level],
                    stride_t=self.strides_t[level],
                    width=cfg.hidden_dim,
                    resnet1d=cfg.resnet1d,
                )
            )

        self.out = nn.Conv1d(cfg.out_dim, cfg.in_dim, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(
            list(zip(list(range(self.levels)), self.downs_t, self.strides_t))
        )
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            if isinstance(stride_t, (tuple, list)) and isinstance(
                down_t, (tuple, list)
            ):
                emb, T = self.output_emb_width, T * np.prod(
                    [s**d for s, d in zip(stride_t, down_t)]
                )
            else:
                emb, T = self.output_emb_width, T * (stride_t**down_t)
            assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        x = self.out(x)
        return x
