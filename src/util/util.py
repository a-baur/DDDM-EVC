from typing import Any, List

import torch
from torch.nn import functional as F


def matmul_with_relative_values(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param x: [b, h, l, m]
    :param y: [h or 1, m, d]
    :return: [b, h, l, d]
    """
    return torch.matmul(x, y.unsqueeze(0))


def matmul_with_relative_keys(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    :param x: [b, h, l, d]
    :param y: [h or 1, m, d]
    :return: [b, h, l, m]
    """
    return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))


def convert_pad_shape(pad_shape: Any) -> List[Any]:
    """
    ?

    :param pad_shape: ?
    :return: ?
    """
    _list = pad_shape[::-1]
    pad_shape = [item for sublist in _list for item in sublist]
    return pad_shape


def relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: [b, h, l, 2*l-1]
    :return: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
        :, :, :length, length - 1 :
    ]
    return x_final


def absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: [b, h, l, l]
    :return: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
    x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
    return x_final


def attention_bias_proximal(length: int) -> torch.Tensor:
    """
    Bias for self-attention to encourage attention to close positions.

    :param length: Length of the sequence.
    :return: A Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


@torch.jit.script  # type: ignore
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    n_channels: torch.Tensor,
) -> torch.Tensor:
    """
    Fused operation of adding, tanh, sigmoid and element-wise multiplication.

    :param input_a: Input tensor A
    :param input_b: Input tensor B
    :param n_channels: Number of channels
    :return: Output tensor
    """
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts
