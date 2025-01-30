"""
Utility functions for dealing with audio data.
"""

import numpy as np

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT


def get_yaapt_f0(audio: np.ndarray, sr: int = 16000, interp: bool = False):
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
        pitch = pYAAPT.yaapt(
            basic.SignalObj(y_pad, sr),
            frame_length=20.0,
            frame_space=5.0,
            nccf_thresh1=0.25,
            tda_frame_length=25.0,
        )
        f0s.append(
            pitch.samp_interp[None, None, :]
            if interp
            else pitch.samp_values[None, None, :]
        )

    return np.vstack(f0s)
