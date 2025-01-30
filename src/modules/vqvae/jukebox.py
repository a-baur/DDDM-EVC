# Adapted from https://github.com/openai/jukebox
from typing import Any

import numpy as np
import torch
import torch.nn as nn

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
        depth: int,
        m_conv: float,
        dilation_growth_rate: int = 1,
        dilation_cycle: Any = None,
        zero_out: bool = False,
        res_scale: bool = False,
    ) -> None:
        super().__init__()
        blocks = []

        if not isinstance(stride_t, (tuple, list)):
            stride_t, down_t = [stride_t], [down_t]

        for k, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
            first_block = k == 0

            filter_t, pad_t = _get_filter_pad(s_t)

            if d_t == 0:
                continue

            for i in range(d_t):
                block = nn.Sequential(
                    nn.Conv1d(
                        input_emb_width if i == 0 and first_block else width,
                        width,
                        filter_t,
                        s_t,
                        pad_t,
                    ),
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        zero_out,
                        res_scale,
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
        depth: int,
        m_conv: float,
        dilation_growth_rate: int = 1,
        dilation_cycle: Any = None,
        zero_out: bool = False,
        res_scale: bool = False,
        reverse_decoder_dilation: bool = False,
        checkpoint_res: bool = False,
    ):
        super().__init__()
        blocks = []

        if not isinstance(stride_t, (tuple, list)):
            stride_t, down_t = [stride_t], [down_t]

        block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
        blocks.append(block)

        for k, (s_t, d_t) in enumerate(zip(stride_t, down_t)):
            last_block = k == len(stride_t) - 1
            if d_t == 0:
                continue

            filter_t, pad_t = _get_filter_pad(s_t)

            for i in range(d_t):
                block = nn.Sequential(
                    Resnet1D(
                        width,
                        depth,
                        m_conv,
                        dilation_growth_rate,
                        dilation_cycle,
                        zero_out=zero_out,
                        res_scale=res_scale,
                        reverse_dilation=reverse_decoder_dilation,
                        checkpoint_res=checkpoint_res,
                    ),
                    nn.ConvTranspose1d(
                        width,
                        input_emb_width if i == (d_t - 1) and last_block else width,
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
    def __init__(
        self,
        input_emb_width: int,
        output_emb_width: int,
        levels: int,
        downs_t: list,
        strides_t: list,
        **block_kwargs: dict,
    ) -> None:
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']

        self.level_blocks = nn.ModuleList()
        for level in range(self.levels):
            stride_t, down_t = self.strides_t[level], self.downs_t[level]
            block = EncoderConvBlock(
                input_emb_width if level == 0 else output_emb_width,
                output_emb_width,
                down_t=down_t,
                stride_t=stride_t,
                **block_kwargs_copy,
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
            if isinstance(stride_t, (tuple, list)):
                emb = self.output_emb_width
                T = T // np.prod([s**d for s, d in zip(stride_t, down_t)])
            else:
                emb, T = self.output_emb_width, T // (stride_t**down_t)
            assert_shape(x, (N, emb, T))

            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(
        self,
        input_emb_width,
        output_emb_width,
        levels,
        downs_t,
        strides_t,
        **block_kwargs,
    ):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        level_block = lambda level, down_t, stride_t: DecoderConvBock(
            output_emb_width, output_emb_width, down_t, stride_t, **block_kwargs
        )
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

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
            if type(stride_t) is tuple or type(stride_t) is list:
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
