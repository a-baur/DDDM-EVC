import torch
from torch import nn
from torch.nn import functional as F


class Mish(nn.Module):
    """
    Mish activation function.

    f(x) = x * tanh(softplus(x))

    Reference:
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/vc/arxiv/papers/1908/1908.08681v1.pdf
    """

    def __init__(self) -> None:
        super(Mish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))
