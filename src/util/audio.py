"""
Utility functions for dealing with audio data.
"""

import math
import random

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
import numpy as np
import parselmouth
import pyworld as pw
import scipy
import torch
import torchaudio.functional as AF


def _random_ratio(min_val: float, max_val: float) -> float:
    ratio = random.uniform(min_val, max_val)
    use_reciprocal = random.uniform(-1, 1) > 0
    if use_reciprocal:
        ratio = 1 / ratio
    return ratio


def get_yaapt_f0(
    audio: np.ndarray, sr: int = 16000, interp: bool = False, framework: str = "pyaapt"
) -> np.ndarray:
    """
    Get the fundamental frequency using YAAPT.

    :param audio: Audio waveform.
    :param sr: Sampling rate.
    :param interp: Interpolate the pitch values.
    :param framework: Framework to use for pitch extraction.
    :return: Fundamental frequency values.
    """
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        try:  # TODO remove
            if framework == "pyaapt":
                pitch = pYAAPT.yaapt(
                    basic.SignalObj(y_pad, sr),
                    frame_length=20.0,
                    frame_space=5.0,
                    nccf_thresh1=0.25,
                    tda_frame_length=25.0,
                ).samp_values
            elif framework == "parselmouth":
                pitch = parselmouth.praat.call(
                    parselmouth.Sound(y_pad, sampling_frequency=sr),
                    "To Pitch",
                    0.005,
                    75,
                    600,
                ).selected_array["frequency"]
            else:
                raise NotImplementedError(f"Unknown framework: {framework}")

            f0s.append(pitch[None, None, :])
        except IndexError:
            f0s.append(np.zeros((1, 1, y_pad.shape[0])))

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


def get_normalized_f0(
    x: torch.Tensor, sr: int = 16000, framework: str = "pyaapt"
) -> torch.Tensor:
    """
    Get the normalized fundamental frequency values.

    :param x: Audio tensor (B x T).
    :param sr: Sampling rate.
    :param framework: Framework to use for pitch extraction.
    :return: Normalized fundamental frequency values.
    """
    f0 = get_yaapt_f0(x.cpu().numpy(), sr, framework=framework)
    f0 = normalize_f0(f0)
    return torch.FloatTensor(f0).to(x.device)


class PraatProcessor:
    """
    Praat processor for audio manipulation.

    Adapted from the unofficial NANSY implementation:
    https://github.com/dhchoi99/NANSY/blob/master/datasets/functional.py

    Use to apply information perturbation to remove
    pitch and speaker related features from the audio.
    Refer to original NANSY paper for more details:
    https://arxiv.org/pdf/2110.14513#appendix.1.A

    Args:
        sample_rate: Sampling rate.
    """

    _DEFAULT_PITCHMEDIAN = 0.0
    _DEFAULT_FORMANTSHIFTRATIO = 1.0
    _DEFAULT_PITCHSHIFTRATIO = 1.0
    _DEFAULT_PITCHRANGERATIO = 1.0

    def __init__(self, sample_rate: int = 16000, flatten_pitch: bool = False) -> None:
        self.sample_rate = sample_rate
        self.parametric_equalizer = ParametricEqualizer(sample_rate)
        self.flatten_pitch = flatten_pitch

    def g_batched(self, audio: torch.Tensor) -> torch.Tensor:
        """Batched version of g"""
        audio = self.parametric_equalizer.apply(audio)
        batch_audio = [self.g(a) for a in audio]
        return torch.stack(batch_audio).to(audio.device)

    def f_batched(self, audio: torch.Tensor) -> torch.Tensor:
        """Batched version of f"""
        audio = self.parametric_equalizer.apply(audio)
        batch_audio = [self.f(a) for a in audio]
        return torch.stack(batch_audio).to(audio.device)

    def g(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove speaker related features from the audio

        g(x) = fs(peq(x))
        """
        sound = audio.numpy().astype(np.float64)

        if self.flatten_pitch:
            sound = self.apply_pitch_flattening(sound)

        sound = self.wav_to_Sound(sound)
        sound = self.formant_shift(sound)

        audio = torch.from_numpy(sound.values).float().squeeze(0)
        return audio

    def f(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove pitch and speaker related features from the audio

        f(x) = fs(pr(peq(x)))
        """
        sound = audio.numpy().astype(np.float64)

        if self.flatten_pitch:
            sound = self.apply_pitch_flattening(sound)

        sound = self.wav_to_Sound(sound)
        sound = self.formant_and_pitch_shift(sound)

        audio = torch.from_numpy(sound.values).float().squeeze(0)
        return audio

    def formant_and_pitch_shift(self, sound: parselmouth.Sound) -> parselmouth.Sound:
        """Apply formant shifting and pitch randomization

        fs(pr(x))
        """
        formant_shifting_ratio = _random_ratio(1, 1.4)
        pitch_shift_ratio = _random_ratio(1, 2)
        pitch_range_ratio = _random_ratio(1, 1.5)

        sound_new = self.apply_formant_and_pitch_shift(
            sound,
            formant_shift_ratio=formant_shifting_ratio,
            pitch_shift_ratio=pitch_shift_ratio,
            pitch_range_ratio=pitch_range_ratio,
        )
        return sound_new

    def formant_shift(self, sound: parselmouth.Sound) -> parselmouth.Sound:
        """Apply formant shifting

        fs(x)
        """
        formant_shifting_ratio = _random_ratio(1, 1.4)

        return self.apply_formant_and_pitch_shift(
            sound,
            formant_shift_ratio=formant_shifting_ratio,
            pitch_shift_ratio=self._DEFAULT_PITCHSHIFTRATIO,
            pitch_range_ratio=self._DEFAULT_PITCHRANGERATIO,
        )

    def wav_to_Sound(
        self, wav: parselmouth.Sound | np.ndarray | list
    ) -> parselmouth.Sound:
        """Load wav file to parselmouth Sound file"""
        if isinstance(wav, parselmouth.Sound):
            sound = wav
        elif isinstance(wav, np.ndarray):
            sound = parselmouth.Sound(
                wav,
                sampling_frequency=self.sample_rate,  # type: ignore
            )
        elif isinstance(wav, list):
            wav_np = np.asarray(wav)
            sound = parselmouth.Sound(
                np.asarray(wav_np),
                sampling_frequency=self.sample_rate,  # type: ignore
            )
        else:
            raise NotImplementedError
        return sound

    def wav_to_Tensor(self, wav: parselmouth.Sound | np.ndarray | list) -> torch.Tensor:
        """Convert wav to torch tensor"""
        if isinstance(wav, np.ndarray):
            wav_tensor = torch.from_numpy(wav)
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav
        elif isinstance(wav, parselmouth.Sound):
            wav_np = wav.values
            wav_tensor = torch.from_numpy(wav_np)
        else:
            raise NotImplementedError
        return wav_tensor

    def get_pitch_median(self, wav: parselmouth.Sound) -> tuple:
        """Get pitch median from wav file"""
        sound = self.wav_to_Sound(wav)
        pitch: parselmouth.Pitch = parselmouth.praat.call(  # type: ignore
            sound, "To Pitch", 0.8 / 75, 75, 600
        )
        pitch_median: float = parselmouth.praat.call(  # type: ignore
            pitch, "Get quantile", 0.0, 0.0, 0.5, "Hertz"
        )
        return pitch, pitch_median

    def apply_formant_and_pitch_shift(
        self,
        sound: parselmouth.Sound,
        formant_shift_ratio: float,
        pitch_shift_ratio: float,
        pitch_range_ratio: float,
    ) -> parselmouth.Sound:
        """
        Use praat 'Change Gender' backend to manipulate
        pitch and formant 'Change Gender' function:
        https://www.fon.hum.uva.nl/praat/manual/Sound__Change_gender___.html
        """
        if pitch_shift_ratio != 1.0:
            pitch, pitch_median = self.get_pitch_median(sound)
            new_pitch_median = pitch_median * pitch_shift_ratio

            # https://github.com/praat/praat/issues/1926#issuecomment-974909408
            pitch_minimum: float = parselmouth.praat.call(  # type: ignore
                pitch, "Get minimum", 0.0, 0.0, "Hertz", "Parabolic"
            )
            newMedian = pitch_median * pitch_shift_ratio
            scaledMinimum = pitch_minimum * pitch_shift_ratio
            resultingMinimum = (
                newMedian + (scaledMinimum - newMedian) * pitch_range_ratio
            )
            if resultingMinimum < 0:
                new_pitch_median = self._DEFAULT_PITCHMEDIAN
                pitch_range_ratio = self._DEFAULT_PITCHRANGERATIO

            if math.isnan(new_pitch_median):
                new_pitch_median = self._DEFAULT_PITCHMEDIAN
                pitch_range_ratio = self._DEFAULT_PITCHRANGERATIO
        else:
            new_pitch_median = self._DEFAULT_PITCHMEDIAN
            pitch = sound.to_pitch()

        new_sound: parselmouth.Sound = parselmouth.praat.call(  # type: ignore
            (sound, pitch),  # type: ignore
            "Change gender",
            formant_shift_ratio,
            new_pitch_median,
            pitch_range_ratio,
            1.0,
        )

        return new_sound

    def apply_pitch_flattening(self, wav: np.ndarray) -> np.ndarray:
        """
        Flatten the pitch of the audio signal.

        :param wav: Audio waveform.
        :return: Flattened audio waveform.
        """
        _f0, t = pw.dio(wav, self.sample_rate)
        f0 = pw.stonemask(wav, _f0, t, self.sample_rate)
        sp = pw.cheaptrick(wav, f0, t, self.sample_rate)
        ap = pw.d4c(wav, f0, t, self.sample_rate)

        mean_f0 = np.mean(f0[f0 > 0])
        flat_f0 = np.where(f0 > 0, mean_f0, 0.0)

        y = pw.synthesize(flat_f0, sp, ap, self.sample_rate)
        return y[: wav.shape[0]]  # Ensure the output shape matches the input shape


class ParametricEqualizer:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def apply(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply parametric equalizer for random frequency shaping

        peq(x)
        """
        cutoff_low_freq = 60.0
        cutoff_high_freq = 10000.0

        q_min = 2
        q_max = 5

        num_filters = 8 + 2  # 8 for peak, 2 for high/low
        key_freqs = [
            self._power_ratio(float(z) / num_filters, cutoff_low_freq, cutoff_high_freq)
            for z in range(num_filters)
        ]
        Qs = [
            self._power_ratio(random.uniform(0, 1), q_min, q_max)
            for _ in range(num_filters)
        ]
        gains = [random.uniform(-12, 12) for _ in range(num_filters)]

        # peak filters
        for i in range(1, 9):
            audio = self._apply_iir_filter(
                audio,
                ftype="peak",
                dBgain=gains[i],
                cutoff_freq=key_freqs[i],
                Q=Qs[i],
            )

        # high-shelving filter
        audio = self._apply_iir_filter(
            audio,
            ftype="high",
            dBgain=gains[-1],
            cutoff_freq=key_freqs[-1],
            Q=Qs[-1],
        )

        # low-shelving filter
        audio = self._apply_iir_filter(
            audio,
            ftype="low",
            dBgain=gains[0],
            cutoff_freq=key_freqs[0],
            Q=Qs[0],
        )

        return audio

    def _apply_iir_filter(
        self,
        audio: torch.Tensor,
        ftype: str,
        dBgain: float,
        cutoff_freq: float,
        Q: float,
        torch_backend: bool = True,
    ) -> torch.Tensor:
        """Apply IIR filter to audio tensor for parametric equalization.

        Implemented using the cookbook https://webaudio.github.io/Audio-EQ-Cookbook/audio-eq-cookbook.html
        """
        if ftype == "low":
            b0, b1, b2, a0, a1, a2 = self._low_shelf_coeffs(dBgain, cutoff_freq, Q)
        elif ftype == "high":
            b0, b1, b2, a0, a1, a2 = self._high_shelf_coeffs(dBgain, cutoff_freq, Q)
        elif ftype == "peak":
            b0, b1, b2, a0, a1, a2 = self._peaking_coeffs(dBgain, cutoff_freq, Q)
        else:
            raise NotImplementedError
        if torch_backend:
            audio_out = AF.biquad(audio, b0, b1, b2, a0, a1, a2)
        else:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter_zi.html
            audio_np = audio.numpy()
            b = np.asarray([b0, b1, b2])
            a = np.asarray([a0, a1, a2])
            zi = scipy.signal.lfilter_zi(b, a) * audio_np[0]
            audio_out, _ = scipy.signal.lfilter(b, a, audio_np, zi=zi)
            audio_out = torch.from_numpy(audio_out)
        return audio_out

    def _low_shelf_coeffs(self, dBgain: float, cutoff_freq: float, Q: float) -> tuple:
        A = math.pow(10, dBgain / 40.0)

        w0 = 2 * math.pi * cutoff_freq / self.sample_rate
        alpha = math.sin(w0) / 2 / Q
        # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

        b0 = A * ((A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

        a0 = (A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * math.cos(w0))
        a2 = (A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
        return b0, b1, b2, a0, a1, a2

    def _high_shelf_coeffs(self, dBgain: float, cutoff_freq: float, Q: float) -> tuple:
        A = math.pow(10, dBgain / 40.0)

        w0 = 2 * math.pi * cutoff_freq / self.sample_rate
        alpha = math.sin(w0) / 2 / Q
        # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

        b0 = A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * math.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)

        a0 = (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * math.cos(w0))
        a2 = (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha
        return b0, b1, b2, a0, a1, a2

    def _peaking_coeffs(self, dBgain: float, cutoff_freq: float, Q: float) -> tuple:
        A = math.pow(10, dBgain / 40.0)

        w0 = 2 * math.pi * cutoff_freq / self.sample_rate
        alpha = math.sin(w0) / 2 / Q
        # alpha = alpha / math.sqrt(2) * math.sqrt(A + 1 / A)

        b0 = 1 + alpha * A
        b1 = -2 * math.cos(w0)
        b2 = 1 - alpha * A

        a0 = 1 + alpha / A
        a1 = -2 * math.cos(w0)
        a2 = 1 - alpha / A
        return b0, b1, b2, a0, a1, a2

    @staticmethod
    def _power_ratio(r: float, a: float, b: float) -> float:
        return a * math.pow((b / a), r)
