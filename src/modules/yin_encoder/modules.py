import math
from typing import Iterable

import numpy as np
import torch

import util

from .yin import (
    cumulativeMeanNormalizedDifferenceFunctionTorch,
    differenceFunctionTorch,
)


class YINTransform_(torch.nn.Module):
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


#
#
# import torch
# import torch.nn.functional as F
#
#
# def differenceFunction(x, N, tau_max):
#     """
#     Compute difference function of data x. This corresponds to equation (6) in [1]
#     This solution is implemented directly with torch rfft.
#
#
#     :param x: audio data (Tensor)
#     :param N: length of data
#     :param tau_max: integration window size
#     :return: difference function
#     :rtype: list
#     """
#
#     # x = np.array(x, np.float64) #[B,T]
#     assert x.dim() == 2
#     b, w = x.shape
#     if w < tau_max:
#         x = F.pad(
#             x,
#             (tau_max - w - (tau_max - w) // 2, (tau_max - w) // 2),
#             "constant",
#             mode="reflect",
#         )
#     w = tau_max
#     # x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
#     x_cumsum = torch.cat(
#         [torch.zeros([b, 1], device=x.device), (x * x).cumsum(dim=1)], dim=1
#     )
#     size = w + tau_max
#     p2 = (size // 32).bit_length()
#     # p2 = ceil(log2(size+1 // 32))
#     nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
#     size_pad = min(n * 2**p2 for n in nice_numbers if n * 2**p2 >= size)
#     fc = torch.fft.rfft(x, size_pad)  # [B,F]
#     conv = torch.fft.irfft(fc * fc.conj())[:, :tau_max]
#     return (
#         x_cumsum[:, w : w - tau_max : -1]
#         + x_cumsum[:, w]
#         - x_cumsum[:, :tau_max]
#         - 2 * conv
#     )
#
#
# def differenceFunction_np(x, N, tau_max):
#     """
#     Compute difference function of data x. This corresponds to equation (6) in [1]
#     This solution is implemented directly with Numpy fft.
#
#
#     :param x: audio data
#     :param N: length of data
#     :param tau_max: integration window size
#     :return: difference function
#     :rtype: list
#     """
#
#     x = np.array(x, np.float64)
#     w = x.size
#     tau_max = min(tau_max, w)
#     x_cumsum = np.concatenate((np.array([0.0]), (x * x).cumsum()))
#     size = w + tau_max
#     p2 = (size // 32).bit_length()
#     nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
#     size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)
#     fc = np.fft.rfft(x, size_pad)
#     conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
#     return x_cumsum[w : w - tau_max : -1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv
#
#
# def cumulativeMeanNormalizedDifferenceFunction(df, N, eps=1e-8):
#     """
#     Compute cumulative mean normalized difference function (CMND).
#
#     This corresponds to equation (8) in [1]
#
#     :param df: Difference function
#     :param N: length of data
#     :return: cumulative mean normalized difference function
#     :rtype: list
#     """
#     # np.seterr(divide='ignore', invalid='ignore')
#     # scipy method, assert df>0 for all element
#     #   cmndf = df[1:] * np.asarray(list(range(1, N))) / (np.cumsum(df[1:]).astype(float) + eps)
#     B, _ = df.shape
#     cmndf = (
#         df[:, 1:]
#         * torch.arange(1, N, device=df.device, dtype=df.dtype).view(1, -1)
#         / (df[:, 1:].cumsum(dim=-1) + eps)
#     )
#     return torch.cat(
#         [torch.ones([B, 1], device=df.device, dtype=df.dtype), cmndf], dim=-1
#     )
#
#
# def differenceFunctionTorch(xs: torch.Tensor, N, tau_max) -> torch.Tensor:
#     """pytorch backend batch-wise differenceFunction
#     has 1e-4 level error with input shape of (32, 22050*1.5)
#     Args:
#         xs:
#         N:
#         tau_max:
#
#     Returns:
#
#     """
#     xs = xs.double()
#     w = xs.shape[-1]
#     tau_max = min(tau_max, w)
#     zeros = torch.zeros((xs.shape[0], 1))
#     x_cumsum = torch.cat(
#         (
#             torch.zeros((xs.shape[0], 1), device=xs.device),
#             (xs * xs).cumsum(dim=-1, dtype=torch.double),
#         ),
#         dim=-1,
#     )  # B x w
#     size = w + tau_max
#     p2 = (size // 32).bit_length()
#     nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
#     size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)
#
#     fcs = torch.fft.rfft(xs, n=size_pad, dim=-1)
#     convs = torch.fft.irfft(fcs * fcs.conj())[:, :tau_max]
#     y1 = torch.flip(x_cumsum[:, w - tau_max + 1 : w + 1], dims=[-1])
#     y = y1 + x_cumsum[:, w].unsqueeze(-1) - x_cumsum[:, :tau_max] - 2 * convs
#     return y
#
#
# def cumulativeMeanNormalizedDifferenceFunctionTorch(
#     dfs: torch.Tensor, N, eps=1e-8
# ) -> torch.Tensor:
#     arange = torch.arange(1, N, device=dfs.device, dtype=torch.float64)
#     cumsum = torch.cumsum(dfs[:, 1:], dim=-1, dtype=torch.float64).to(dfs.device)
#
#     cmndfs = dfs[:, 1:] * arange / (cumsum + eps)
#     cmndfs = torch.cat(
#         (torch.ones(cmndfs.shape[0], 1, device=dfs.device), cmndfs), dim=-1
#     )
#     return cmndfs
#
#
# class YINTransform(torch.nn.Module):
#     def __init__(
#         self,
#         sample_rate=16000,
#         hop_length=320,
#         win_length=2048,
#         tau_max=2048,
#         midi_start=5,
#         midi_end=85,
#         octave_range=12,
#     ):
#         super(YINTransform, self).__init__()
#         self.sr = sample_rate
#         self.w_step = hop_length
#         self.W = win_length
#         self.tau_max = tau_max
#         self.unfold = torch.nn.Unfold((1, self.W), 1, 0, stride=(1, self.w_step))
#         midis = list(range(midi_start, midi_end))
#         self.len_midis = len(midis)
#         c_ms = torch.tensor([self.midi_to_lag(m, octave_range) for m in midis])
#         # self.register_buffer("c_ms", c_ms)
#         # self.register_buffer("c_ms_ceil", torch.ceil(self.c_ms).long())
#         # self.register_buffer("c_ms_floor", torch.floor(self.c_ms).long())
#         self.c_ms = c_ms
#         self.c_ms_ceil = torch.ceil(self.c_ms).long()
#         self.c_ms_floor = torch.floor(self.c_ms).long()
#
#     def midi_to_lag(self, m: int, octave_range: float = 12):
#         """converts midi-to-lag, eq. (4)
#
#         Args:
#             m: midi
#             sr: sample_rate
#             octave_range:
#
#         Returns:
#             lag: time lag(tau, c(m)) calculated from midi, eq. (4)
#
#         """
#         f = 440 * math.pow(2, (m - 69) / octave_range)
#         lag = self.sr / f
#         return lag
#
#     def yingram_from_cmndf(self, cmndfs: torch.Tensor) -> torch.Tensor:
#         """yingram calculator from cMNDFs(cumulative Mean Normalized Difference Functions)
#
#         Args:
#             cmndfs: torch.Tensor
#                 calculated cumulative mean normalized difference function
#                 for details, see models/yin.py or eq. (1) and (2)
#             ms: list of midi(int)
#             sr: sampling rate
#
#         Returns:
#             y:
#                 calculated batch yingram
#
#
#         """
#         self.c_ms = self.c_ms.to(cmndfs.device)
#         self.c_ms_ceil = self.c_ms_ceil.to(cmndfs.device)
#         self.c_ms_floor = self.c_ms_floor.to(cmndfs.device)
#
#         y = (cmndfs[:, self.c_ms_ceil] - cmndfs[:, self.c_ms_floor]) / (
#             self.c_ms_ceil - self.c_ms_floor
#         )
#         y = (
#             y.unsqueeze(0) * (self.c_ms - self.c_ms_floor).unsqueeze(0)
#             + cmndfs[:, self.c_ms_floor]
#         )
#         return y
#
#     def yingram(self, x: torch.Tensor):
#         """calculates yingram from raw audio (multi segment)
#
#         Args:
#             x: raw audio, torch.Tensor of shape (t)
#
#         Returns:
#             yingram: yingram. torch.Tensor of shape (80 x t')
#
#         """
#         # x.shape: t -> B,T, B,T = x.shape
#         B, T = x.shape
#
#         frames = self.unfold(x.view(B, 1, 1, T))
#         frames = frames.permute(0, 2, 1).contiguous().view(-1, self.W)  # [B* frames, W]
#         # If not using gpu, or torch not compatible, implemented numpy batch function is still fine
#         dfs = differenceFunctionTorch(frames, frames.shape[-1], self.tau_max).to(
#             x.device
#         )
#         cmndfs = cumulativeMeanNormalizedDifferenceFunctionTorch(dfs, self.tau_max).to(
#             x.device
#         )
#         yingram = self.yingram_from_cmndf(cmndfs).to(x.device)  # [B*frames,F]
#         yingram = yingram.view(B, -1, self.len_midis).permute(0, 2, 1)  # [B,F,T]
#         return yingram.float()
#
#
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
