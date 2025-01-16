"""
Utility functions for dealing with sqeuential data.
"""
import torch


def sequence_mask(length: torch.Tensor, max_length: int = None) -> torch.Tensor:
    """
    Create a boolean mask to ignore the padding
    elements in a batch of sequences.

    :example:
    >>> length = torch.tensor([3, 2, 1])
    >>> sequence_mask(length, max_length=3)
    tensor([[ True,  True,  True],
            [ True,  True, False],
            [ True, False, False]])

    :param length: Length of unpadded sequence (B,).
    :param max_length: Maximum length of the sequences.
    :return: Boolean mask (B, T).
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def temporal_avg_pool(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Apply average pooling to the sequence data.

    Use sequence mask so that padding values are
    not considered in the average pooling.

    :param x: Input tensor (B, C, T).
    :param mask: Sequence mask for padding values (B, T).
    :return: Output tensor (B, C).
    """
    if mask is None:
        out = torch.mean(x, dim=2)
    else:
        len_ = mask.sum(dim=2)
        x = x.sum(dim=2)
        out = torch.div(x, len_)

    return out
