import math
from typing import Iterable

import numpy as np
import torch

import util

from .yin import (
    cumulativeMeanNormalizedDifferenceFunctionTorch,
    differenceFunctionTorch,
)


class YINTransform(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        win_length: int = 1480,
        hop_length: int = 320,
        tau_max: int = 1480,
        semitone_range: float = 12,
    ):
        super().__init__()
        self.sr = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.tau_max = tau_max
        self.semitone_range = semitone_range

    def get_midi_range(self) -> range:
        """get midi range

        Returns:
            midi_range: midi range

        """
        m_min = math.ceil(
            69 + self.semitone_range * math.log2(self.sr / (self.tau_max * 440))
        )
        m_max = 85  # Or another sensible upper bound based on your use case
        return range(max(m_min, 5), m_max)  # Ensure m_min >= 5

    def midi_to_lag(self, m: int) -> float:
        """converts midi-to-lag, eq. (4)

        Args:
            m: midi

        Returns:
            lag: time lag(tau, c(m)) calculated from midi, eq. (4)

        """
        f = 440 * math.pow(2, (m - 69) / self.semitone_range)
        lag = self.sr / f
        return lag

    def yingram_from_cmndf(self, cmndfs: torch.Tensor, ms: Iterable) -> torch.Tensor:
        """yingram calculator from cMNDFs(cumulative Mean Normalized Difference Functions)

        Args:
            cmndfs: torch.Tensor
                calculated cumulative mean normalized difference function
                for details, see models/yin.py or eq. (1) and (2)
            ms: list of midi(int)

        Returns:
            y:
                calculated batch yingram


        """
        c_ms = np.asarray([self.midi_to_lag(m) for m in ms])
        c_ms = torch.from_numpy(c_ms).to(cmndfs.device)
        c_ms_ceil = torch.ceil(c_ms).long().to(cmndfs.device)
        c_ms_floor = torch.floor(c_ms).long().to(cmndfs.device)

        y = (cmndfs[:, c_ms_ceil] - cmndfs[:, c_ms_floor]) / (
            c_ms_ceil - c_ms_floor
        ).unsqueeze(0) * (c_ms - c_ms_floor).unsqueeze(0) + cmndfs[:, c_ms_floor]
        return y

    def yingram(self, x: torch.Tensor) -> torch.Tensor:
        """calculates yingram from raw audio (multi segment)

        Args:
            x: raw audio, torch.Tensor of shape (t)

        Returns:
            yingram: yingram. torch.Tensor of shape (80 x t')

        """
        startFrames = range(0, x.shape[-1] - self.hop_length, self.hop_length)
        # times = startFrames / sr
        frames = [x[..., t : t + self.win_length] for t in startFrames]
        # padding
        frames = util.pad_tensors_to_length(frames, self.win_length)
        frames_torch = torch.stack(frames, dim=0).to(x.device)

        # If not using gpu, or torch not compatible, implemented numpy batch function is still fine
        dfs = differenceFunctionTorch(frames_torch, self.tau_max)
        cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, self.tau_max)

        midis = self.get_midi_range()
        yingram = self.yingram_from_cmndf(cmndfs, midis)
        return yingram

    def yingram_batch(self, x: torch.Tensor) -> torch.Tensor:
        """calculates yingram from batch raw audio.
        currently calculates batch-wise through for loop, but seems it can be implemented to act batch-wise

        Args:
            x: raw audio, torch.Tensor of shape (B x t)

        Returns:
            yingram: yingram. torch.Tensor of shape (B x 80 x t')

        """
        batch_results = []
        for i in range(len(x)):
            yingram = self.yingram(x[i])
            batch_results.append(yingram)
        result = torch.stack(batch_results, dim=0).float()
        result = result.permute((0, 2, 1)).to(x.device)
        return result


# if __name__ == "__main__":
#     import torch
#     import matplotlib.pyplot as plt
#
#     wav, sr = torchaudio.load(util.get_root_path() / "sample/src.wav")
#     wav = wav[:, :38000]
#     #    wav = torch.randn(1,40965)
#
#     wav = torch.nn.functional.pad(wav, (0, (-wav.shape[1]) % 256))
#     pitch = YINTransform(sample_rate=sr)
#
#     with torch.no_grad():
#         ps = pitch.yingram(wav)
#         print(ps.shape)
#         plt.figure(figsize=(15, 10))
#         plt.subplot(2, 1, 1)
#         plt.pcolor(ps[0].numpy(), cmap="magma")
#         plt.colorbar()
#         plt.subplot(2, 1, 2)
#         plt.pcolor(ps[0][15:65, :].numpy(), cmap="magma")
#         plt.colorbar()
#         plt.tight_layout()
#
#         plt.show()
