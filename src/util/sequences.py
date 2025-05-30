"""
Utility functions for dealing with sequential data.
"""

import torch
from torch.nn import functional as F


def sequence_mask(
    length: torch.Tensor, max_length: int = None, add_channel_dim: bool = True
) -> torch.Tensor:
    """
    Create a boolean mask to ignore the padding
    elements in a batch of sequences.

    :example:
    >>> length = torch.tensor([3, 2, 1])
    >>> sequence_mask(length, max_length=3, add_channel_dim=False)
    tensor([[ True,  True,  True],
            [ True,  True, False],
            [ True, False, False]])

    :param length: Length of unpadded sequence (B,).
    :param max_length: Maximum length of the sequences.
    :param add_channel_dim: Whether to add channel dimension. (B, T) -> (B, 1, T).
    :return: Boolean mask (B, T).
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    mask = x.unsqueeze(0) < length.unsqueeze(1)
    if add_channel_dim:
        return mask.unsqueeze(1)
    else:
        return mask


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


def convert_pad_shape(pad_shape: list[list]) -> list[list]:
    """
    Reverses order and flattens the pad_shape
    for F.pad.


    >>> convert_pad_shape([[1, 2], [3, 4]])
    [3, 4, 1, 2]

    :param pad_shape: Padding per dimension.
    :return: Flattened and reversed pad_shape for F.pad
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


def get_conv_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    Calculate padding size for convolutional layers.

    :param kernel_size: Kernel size of the convolutional layer
    :param dilation: Dilation rate of the convolutional layer
    :return: Padding size
    """
    return int((kernel_size * dilation - dilation) / 2)


def pad_tensors_to_length(
    xs: list[torch.Tensor], length: int, value: int = None
) -> list[torch.Tensor]:
    """Pad the input tensors to the given length."""
    pad_amount = [length - x.shape[-1] for x in xs]
    return [
        F.pad(x, (0, pad), value=value) if pad > 0 else x
        for x, pad in zip(xs, pad_amount)
    ]


def get_u_net_compatible_length(length: int, num_downsamplings_in_unet: int = 2) -> int:
    """
    Get the nearest length that is compatible with U-Net architecture.

    :param length: Length of the input sequence
    :param num_downsamplings_in_unet: Number of downsamplings in U-Net
    :return: Nearest length compatible with U-Net
    """
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def pad_for_xlsr(x: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Pads the input audio to ensure frame alignment between XLS-R embeddings
    and Mel spectrograms.

    Wav2Vec2/XLS-R uses a 20ms stride and a 25ms window. The padding ensures
    that the last stride of XLS-R embeddings aligns with the last frame of
    the Mel spectrogram.

    :param x: Input audio waveform
    :param sampling_rate: Sampling rate
    :return: Padded audio waveform
    """
    win_length = int(0.025 * sampling_rate)
    hop_length = int(0.020 * sampling_rate)
    return pad_for_mel_alignment(x, win_length, hop_length)


def pad_for_mel_alignment(
    x: torch.Tensor, win_length: int, hop_length: int
) -> torch.Tensor:
    """
    Pad the input tensor to ensure frame alignment with centered mel-spectrogram.

    :param x: Input tensor
    :param win_length: Window length
    :param hop_length: Hop length
    :return: Padded tensor
    """
    pad = round((win_length - hop_length) // 2)
    return F.pad(x, (pad, pad), mode="reflect")


def forward_fill(x: torch.Tensor) -> torch.Tensor:
    """Forward fill zerores in a 2d-tensor

    :param x: Input tensor (B,T)
    :return: Forward fill tensor (B,T)
    """
    n_dim, t_dim = x.shape
    # Generate indices range
    rng = torch.arange(t_dim)

    rng_2d = rng.unsqueeze(0).repeat(n_dim, 1)
    # Replace indices to zero for elements that equal zero
    rng_2d[x == 0] = 0

    # Forward fill of indices range so all zero elements will be replaced with previous non-zero index.
    idx = rng_2d.cummax(1).values
    x = x[torch.arange(n_dim)[:, None], idx]
    return x
