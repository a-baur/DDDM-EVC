import torch
import torchaudio
import torchaudio.functional as AF

import util
from config import load_hydra_config
from data import MelTransform
from modules.yin_encoder.yingram import Yingram


def delta(yingram: torch.Tensor, padding: str = "replicate") -> torch.Tensor:
    delta = yingram[..., 1:] - yingram[..., :-1]

    if padding == "replicate":
        first = delta[..., :1]  # replicate first derivative
    elif padding == "zero":
        first = torch.zeros_like(delta[..., :1])
    else:
        raise ValueError("padding must be 'replicate' or 'zero'")

    delta = torch.cat([first, delta], dim=-1)
    return delta


def inject_noise(
    yingram: torch.Tensor,
    noise_level: float = 0.1,
    noise_type: str = "gaussian",
) -> torch.Tensor:
    if noise_type == "gaussian":
        noise = torch.randn_like(yingram) * noise_level
    elif noise_type == "uniform":
        noise = (torch.rand_like(yingram) - 0.5) * 2 * noise_level
    else:
        raise ValueError("noise_type must be 'gaussian' or 'uniform'")

    return yingram + noise


def resample(
    yingram: torch.Tensor,
    f_orig: int = 50,  # original frame-rate (Hz)
    f_coarse: int = 10,  # desired coarse rate
    lowpass_width: int = 8,
    rolloff: float = 0.95,
) -> torch.Tensor:
    # 1. Down-sample
    y_low = AF.resample(
        yingram,
        orig_freq=f_orig,
        new_freq=f_coarse,
        lowpass_filter_width=lowpass_width,
        rolloff=rolloff,
        resampling_method="sinc_interp_hann",
    )

    # 2. Up-sample back to original rate
    y_blurred = AF.resample(
        y_low,
        orig_freq=f_coarse,
        new_freq=f_orig,
        lowpass_filter_width=lowpass_width,
        rolloff=rolloff,
        resampling_method="sinc_interp_hann",
    )

    # length mismatch (last frame or two); pad/trim
    if y_blurred.size(-1) < yingram.size(-1):
        pad = yingram.size(-1) - y_blurred.size(-1)
        y_blurred = torch.nn.functional.pad(y_blurred, (0, pad))
    else:
        y_blurred = y_blurred[..., : yingram.size(-1)]

    return y_blurred.squeeze(1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    wav, sr = torchaudio.load(util.get_root_path() / "sample/trg.wav")
    wav = wav[:, :38000]
    #    wav = torch.randn(1,40965)
    print(wav.min(), wav.max())

    # Load config and models
    cfg = load_hydra_config("evc_xlsr_yin")
    mel_transform = MelTransform(cfg.data.mel_transform)
    pitch_1 = Yingram(sample_rate=sr)

    with torch.no_grad():
        fig, ax = plt.subplots(5, 1, figsize=(10, 15))

        mel = mel_transform(wav)
        print(mel.shape)
        ax[0].imshow(mel[0, :, :118].numpy(), aspect="auto", origin="lower")
        ax[0].set_title("Mel spectrogram")
        ax[0].set_xlabel("Time (frames)")
        ax[0].set_ylabel("Frequency (bins)")

        semitone_shift = torch.tensor([-10])
        ps = pitch_1(wav, semitone_shift=semitone_shift)
        print(ps.shape)
        ax[1].imshow(ps[0, :, :118].numpy(), aspect="auto", origin="lower")
        ax[1].set_title("Yingram spectrogram")
        ax[1].set_xlabel("Time (frames)")
        ax[1].set_ylabel("Frequency (bins)")

        ps2 = delta(ps)
        print(ps2.shape)
        ax[2].imshow(ps2[0, :, :118].numpy(), aspect="auto", origin="lower")
        ax[2].set_title("Delta Yingram")
        ax[2].set_xlabel("Time (frames)")
        ax[2].set_ylabel("Frequency (bins)")

        ps3 = resample(ps, f_orig=50, f_coarse=10)
        print(ps3.shape)
        ax[3].imshow(ps3[0, :, :118].numpy(), aspect="auto", origin="lower")
        ax[3].set_title("Resampled Yingram")
        ax[3].set_xlabel("Time (frames)")
        ax[3].set_ylabel("Frequency (bins)")

        ps4 = inject_noise(ps, noise_level=0.3, noise_type="gaussian")
        print(ps4.shape)
        ax[4].imshow(ps4[0, :, :118].numpy(), aspect="auto", origin="lower")
        ax[4].set_title("Yingram with noise")
        ax[4].set_xlabel("Time (frames)")
        ax[4].set_ylabel("Frequency (bins)")

        plt.tight_layout()
        plt.show()
