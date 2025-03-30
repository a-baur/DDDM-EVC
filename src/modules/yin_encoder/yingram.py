from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import util
from config import load_hydra_config
from data import MelTransform


def m2l(sr: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Midi-to-lag converter.
    Args:
        sr: sample rate.
        m: midi-scale.
    Returns;
        corresponding time-lag.
    """
    return sr / (440 * 2 ** ((m - 69) / 12))


def l2m(sr: float, lag: float) -> float:
    """Lag-to-midi converter.
    Args:
        sr: sample rate.
        lag: time-lag.
    Returns:
        corresponding midi-scale value.
    """
    return 12 * np.log2(sr / (440 * lag)) + 69


class Yingram(nn.Module):
    """Yingram, Midi-scale cumulative mean-normalized difference."""

    def __init__(
        self,
        hop_length: int = 320,  # 16000 / 50
        win_length: int = 1468,  # 16000 / 10.913
        fmin: float = 10.91,  # midi 5
        fmax: float = 1046.50,  # midi 84
        scope_fmin: float = 25.97,  # midi 20
        scope_fmax: float = 440.00,  # midi 69
        bins: int = 20,
        sample_rate: int = 16000,
    ):
        """Initializer.
        Args:
            hop_length: the number of the frames between adjacent windows.
            win_length: width of the window.
            lmin, lmax: bounds of time-lag,
                it could be `sr / fmax` or `sr / fmin`.
            bins: the number of the bins per semitone.
            sample_rate: sample rate, default 16khz.
        """
        super().__init__()
        sr = sample_rate  # alias
        self.strides = hop_length
        self.windows = win_length
        self.bins = bins
        self.sr = sr
        self.lmin = int(sr / fmax)
        self.lmax = int(np.ceil(sr / fmin) + 1)
        # midi range
        self.mmin, self.mmax = Yingram.midi_range(sr, sr / fmax, sr / fmin)

        scope_lmin, scope_lmax = sr / scope_fmax, sr / scope_fmin
        scope_mmin, scope_mmax = Yingram.midi_range(sr, scope_lmin, scope_lmax)
        self.scope_start = (scope_mmin - self.mmin) * bins
        self.scope_range = (scope_mmax - scope_mmin + 1) * bins

    @staticmethod
    def midi_range(sr: int, lmin: int | float, lmax: int | float) -> Tuple[int, int]:
        """Convert time-lag range to midi-range.
        Args:
            sr: sample rate.
            lmin, lmax: bounds of time-lag.
        Returns:
            bounds of midi-scale range, closed interval.
        """
        return round(l2m(sr, lmax)), round(l2m(sr, lmin))

    def forward(
        self, audio: torch.Tensor, semitone_shift: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute the yingram from audio signal.
        Args:
            audio: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            semitone_shift: [torch.float32; [B]], semitone shift.
        Returns:
            [torch.float32; [B, T / `strides`, `bins` x (M - m + 1)]], yingram,
                where M = l2m(`lmin`), m = l2m(`max`)
                      l2m(l) = 12 x log2(`sr` / (440 * l)) + 69
        """
        w, tau_max = self.windows, self.lmax
        # [B, T / strides, windows]

        # pad for mel spectrogram alignment
        audio = util.pad_for_mel_alignment(audio, w, self.strides)

        frames = audio.unfold(-1, w, self.strides)
        # [B, T / strides, windows + 1]
        fft = torch.fft.rfft(frames, w * 2, dim=-1)
        # [B, T / strides, windows], symmetric
        corr = torch.fft.irfft(fft.abs().square(), dim=-1)
        # [B, T / strides, windows + 1]
        cumsum = F.pad(frames.square().cumsum(dim=-1), [1, 0])
        # [B, T / strides, lmax], difference function
        diff = (
            torch.flip(cumsum[..., w - tau_max + 1 : w + 1], dims=[-1])
            - 2 * corr[..., :tau_max]
            + cumsum[..., w, None]
            - cumsum[..., :tau_max]
        )
        # [B, T / strides, lmax - 1]
        cumdiff = diff[..., 1:] / (diff[..., 1:].cumsum(dim=-1) + 1e-7)
        ## in NANSY, Eq(1), it does not normalize the cumulative sum with lag size
        ## , but in YIN, Eq(8), it normalize the sum with their lags
        cumdiff = cumdiff * torch.arange(1, tau_max, device=cumdiff.device)
        # [B, T / strides, lmax], cumulative mean normalized difference
        cumdiff = F.pad(cumdiff, [1, 0], value=1.0)
        # [bins x (mmax - mmin + 1)]
        steps = self.bins**-1
        lags = m2l(
            self.sr,
            torch.arange(self.mmin, self.mmax + 1, step=steps, device=cumdiff.device),
        )
        lceil, lfloor = lags.ceil().long(), lags.floor().long()
        # [B, T / strides, bins x (mmax - mmin + 1)], yingram
        yingram = (cumdiff[..., lceil] - cumdiff[..., lfloor]) * (lags - lfloor) / (
            lceil - lfloor
        ) + cumdiff[..., lfloor]
        yingram = self._sample_scope(yingram, semitone_shift)

        # [B, bins x (mmax - mmin + 1), T / strides]
        return yingram.permute(0, 2, 1)

    def _sample_scope(
        self, x: torch.Tensor, semitone_shift: torch.Tensor = None
    ) -> torch.Tensor:
        """Sample the yingram.
        Args:
            x: [torch.float32; [B, T]], audio signal, [-1, 1]-ranged.
            semitone_shift: [torch.float32; [B]], semitone shift.
        Returns:
            [torch.float32; [B, T / `strides`, `bins` x (M - m + 1)]], yingram,
                where M = l2m(`lmin`), m = l2m(`max`)
                      l2m(l) = 12 x log2(`sr` / (440 * l)) + 69
        """
        idx_start = self.scope_start
        idx_end = idx_start + self.scope_range
        if semitone_shift is not None:
            # raising by a semitone is equivalent to shifting down the scope
            idx_shift = -(semitone_shift * self.bins).int()
            return torch.stack(
                [
                    x[i, :, idx_start + idx : idx_end + idx]
                    for i, idx in enumerate(idx_shift)
                ]
            )
        else:
            return x[..., idx_start:idx_end]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    wav, sr = torchaudio.load(util.get_root_path() / "sample/trg.wav")
    wav = wav[:, :16000]
    #    wav = torch.randn(1,40965)
    print(wav.min(), wav.max())

    pitch_2 = Yingram(sample_rate=sr)

    cfg = load_hydra_config("evc_xlsr_yin")
    mel_transform = MelTransform(cfg.data.mel_transform)

    with torch.no_grad():
        plt.figure(figsize=(15, 15))

        mel = mel_transform(wav)
        print(mel.shape)
        plt.subplot(2, 1, 1)
        plt.pcolor(mel[0].numpy(), cmap="magma")
        plt.colorbar()

        semitone_shift = torch.tensor([0])
        ps = pitch_2(wav, semitone_shift)
        print(ps.shape)
        print(pitch_2.scope_range)
        plt.subplot(2, 1, 2)
        plt.pcolor(ps[0, :, :118].numpy(), cmap="magma")
        plt.colorbar()
        plt.tight_layout()

        plt.show()
