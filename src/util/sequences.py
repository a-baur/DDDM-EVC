"""
Utility functions for dealing with sequential data.
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


def random_segment(
    x: torch.Tensor,
    segment_size: int = 38000,
) -> tuple[torch.Tensor, int]:
    """
    Randomly sample a segment from the input tensor.
    If the input tensor is shorter than the segment size,
    zero-padding is applied.

    :param x: Input tensor (..., T).
    :param segment_size: Size of the segment to sample.
    :return: Tuple of sampled segment and its unpadded size.
    """
    audio_size = x.size(-1)
    if audio_size > segment_size:
        start = torch.randint(
            low=0,
            high=audio_size - segment_size,
            size=(1,),
        ).item()
        x = x[..., start : start + segment_size]  # noqa E203
        audio_size = segment_size
    else:
        padding = (0, segment_size - audio_size) + (0, 0) * (x.dim() - 1)
        x = torch.nn.functional.pad(
            x,
            pad=padding,
            mode="constant",
            value=0,
        )

    return x, audio_size
