import math
from typing import Any, List, Tuple, Union

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


def init_weights(
    m: torch.nn.Module,
    mean: float = 0.0,
    std: float = 0.01,
) -> None:
    """
    Initialize the weights of a module.

    :param m: torch.nn.Module
    :param mean: mean of the normal distribution
    :param std: standard deviation of the normal distribution
    :return: None
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    Get padding size for a given kernel size and dilation.

    :param kernel_size: Size of the kernel
    :param dilation: Dilation factor
    :return: Padding size
    """
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: Any) -> List[Any]:
    """
    ?

    :param pad_shape: ?
    :return: ?
    """
    _list = pad_shape[::-1]
    pad_shape = [item for sublist in _list for item in sublist]
    return pad_shape


def intersperse(lst: list, item: Any) -> list:
    """
    Intersperse an item between elements of a list.

    :param lst: List of elements
    :param item: Item to intersperse
    :return: List with items interspersed
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(
    m_p: torch.Tensor, logs_p: torch.Tensor, m_q: torch.Tensor, logs_q: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL(P||Q) with P and Q being diagonal Gaussian.

    :param m_p: Mean of P
    :param logs_p: Log variance of P
    :param m_q: Mean of Q
    :param logs_q: Log variance of Q
    :return: KL(P||Q)
    """
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape: torch.Size) -> torch.Tensor:
    """
    Sample from the Gumbel distribution, protect from overflows.

    :param shape: Shape of the output tensor
    :return: Sample from the Gumbel distribution
    """
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x: torch.Tensor) -> torch.Tensor:
    """
    Create a random Gumbel noise tensor with the same shape as x.

    :param x: Input tensor
    :return: Random Gumbel noise tensor
    """
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    """
    Slice segments from the batch of input tensors
    along the time dimension.

    :param x: Input tensor batch (B, D, T)
    :param ids_str: Starting indices of the segments (B,)
    :param segment_size: Size of the segments
    :return: Sliced segments (B, D, segment_size)
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def slice_segments_audio(
    x: torch.Tensor, ids_str: torch.Tensor, segment_size: int = 4
) -> torch.Tensor:
    """
    Slice segments from the input tensor along the time dimension.

    :param x: Input tensor (B, T)
    :param ids_str: Starting indices of the segments (B,)
    :param segment_size: Size of the segments
    :return: Sliced segments (B, segment_size)
    """
    ret = torch.zeros_like(x[:, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, idx_str:idx_end]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: int = None, segment_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly slice segments from the input tensor along the time dimension.

    :param x: Input tensor batch (B, D, T)
    :param x_lengths: Lengths of the input sequences (B,)
    :param segment_size: Size of the segments
    :return: Tuple of
        - Randomly sliced segments for each tensor in batch (B, D, segment_size)
        - Starting indices of the segments (B,)
    """
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = ((torch.rand([b]).to(device=x.device) * ids_str_max).clip(0)).to(
        dtype=torch.long
    )
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(
    length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    """
    Get a 1D timing signal of shape (1, channels, length).

    :param length: Length of the signal
    :param channels: Number of channels
    :param min_timescale: Minimum timescale
    :param max_timescale: Maximum timescale
    :return: 1D timing signal
    """
    position = torch.arange(length, dtype=torch.float)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(
    x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> torch.Tensor:
    """
    Add a 1D timing signal to the input tensor.

    :param x: Input tensor
    :param min_timescale: Minimum timescale
    :param max_timescale: Maximum timescale
    :return: Tensor with timing signal added
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(
    x: torch.Tensor,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
    axis: int = 1,
) -> torch.Tensor:
    """
    Concatenate a 1D timing signal to the input tensor.

    :param x: Input tensor
    :param min_timescale: Minimum timescale
    :param max_timescale: Maximum timescale
    :param axis: Axis to concatenate the timing signal
    :return: Tensor with timing signal concatenated
    """
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int) -> torch.Tensor:
    """
    Create a mask to prevent attention to future tokens.
    Only the lower triangular part of the matrix is filled with ones.

    :param length: Length of the sequence
    :return: Mask tensor
    """
    mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
    return mask


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


def shift_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Shift the input tensor along the time dimension.

    :param x: Input tensor
    :return: Shifted tensor
    """
    x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(
    length: torch.Tensor,
    max_length: int = None,
) -> torch.Tensor:
    """
    Create a boolean mask to ignore the padding
    elements in a batch of sequences.

    :param length: Length of unpadded sequence (B,).
    :param max_length: Maximum length of the sequences.
    :return: Boolean mask (B, T).
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Generate a path from the duration and mask.

    :param duration: Duration tensor (B, 1, T_x)
    :param mask: Mask tensor (B, 1, T_y, T_x)
    :return: Path tensor (B, 1, T_y, T_x)
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(
    parameters: Union[torch.Tensor, List[torch.Tensor]],
    clip_value: float,
    norm_type: int = 2,
) -> float:
    """
    Clip the gradients of the parameters in place.

    :param parameters: Tensor or list of tensors
    :param clip_value: Value to clip the gradients
    :param norm_type: Type of the used norm (e.g. 2 for L2 norm)
    :return: Total norm of the gradients before clipping
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda param: param.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
