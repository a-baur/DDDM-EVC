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

    :param length: Length of unpadded sequence of shape (B,).
    :param max_length: Maximum length of the sequences.
    :return: Boolean mask of shape (B, T).
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
