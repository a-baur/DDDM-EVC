# adapted from https://github.com/patriceguyot/Yin
# https://github.com/NVIDIA/mellotron/blob/master/yin.py

import numpy as np
import torch


def differenceFunctionTorch(xs: torch.Tensor, tau_max: int) -> torch.Tensor:
    """pytorch backend batch-wise differenceFunction
    has 1e-4 level error with input shape of (32, 22050*1.5)
    Args:
        xs:
        N:
        tau_max:

    Returns:

    """
    xs = xs.double()
    w = xs.shape[-1]
    tau_max = min(tau_max, w)
    x_cumsum = torch.cat(
        (
            torch.zeros((xs.shape[0], 1), device=xs.device),
            (xs * xs).cumsum(dim=-1, dtype=torch.double),
        ),
        dim=-1,
    )  # B x w
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)

    fcs = torch.fft.rfft(xs, n=size_pad, dim=-1)
    convs = torch.fft.irfft(fcs * fcs.conj())[:, :tau_max]
    y1 = torch.flip(x_cumsum[:, w - tau_max + 1 : w + 1], dims=[-1])
    y = y1 + x_cumsum[:, w, np.newaxis] - x_cumsum[:, :tau_max] - 2 * convs
    return y


def cumulativeMeanNormalizedDifferenceFunctionTorch(
    dfs: torch.Tensor, N: int, eps: float = 1e-8
) -> torch.Tensor:
    arange = torch.arange(1, N, device=dfs.device, dtype=torch.float64)
    cumsum = torch.cumsum(dfs[:, 1:], dim=-1, dtype=torch.float64).to(dfs.device)

    cmndfs = dfs[:, 1:] * arange / (cumsum + eps)
    cmndfs = torch.cat(
        (torch.ones(cmndfs.shape[0], 1, device=dfs.device), cmndfs), dim=-1
    )
    return cmndfs
