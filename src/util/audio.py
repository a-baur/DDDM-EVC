"""
Utility functions for dealing with audio data.
"""

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import torch


def get_yaapt_f0(
    audio: np.ndarray, sr: int = 16000, interp: bool = False
) -> np.ndarray:
    """
    Get the fundamental frequency using YAAPT.

    :param audio: Audio waveform.
    :param sr: Sampling rate.
    :param interp: Interpolate the pitch values.
    :return: Fundamental frequency values.
    """
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        try:  # TODO remove
            pitch = pYAAPT.yaapt(
                basic.SignalObj(y_pad, sr),
                frame_length=20.0,
                frame_space=5.0,
                nccf_thresh1=0.25,
                tda_frame_length=25.0,
            )
        except IndexError:
            pitch = torch.zeros(1, 1, y_pad.shape[0])
        f0s.append(
            pitch.samp_interp[None, None, :]
            if interp
            else pitch.samp_values[None, None, :]
        )

    return np.vstack(f0s)


def normalize_f0(f0: np.ndarray) -> np.ndarray:
    """
    Normalize the fundamental frequency values.

    :param f0: Fundamental frequency values.
    :return: Normalized fundamental frequency values.
    """
    f0 = f0.copy()
    ii = f0 != 0
    f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
    return f0


def get_normalized_f0(x: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """
    Get the normalized fundamental frequency values.

    :param x: Audio tensor (B x T).
    :param sr: Sampling rate.
    :return: Normalized fundamental frequency values.
    """
    f0 = get_yaapt_f0(x.cpu().numpy(), sr)
    f0 = normalize_f0(f0)
    return torch.FloatTensor(f0).to(x.device)
