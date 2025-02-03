import torch


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
