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


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: int
) -> torch.Tensor:
    in_act = input_a + input_b

    # Parallel execution using jit.fork
    future_tanh = torch.jit.fork(torch.tanh, in_act[:, :n_channels, :])
    future_sigmoid = torch.jit.fork(torch.sigmoid, in_act[:, n_channels:, :])

    t_act = torch.jit.wait(future_tanh)
    s_act = torch.jit.wait(future_sigmoid)

    return t_act * s_act
