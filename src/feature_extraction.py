from pathlib import Path

import torch
import torchaudio

import util
from config import load_hydra_config
from data import MelTransform

AUDIO_PATH = Path(r"D:\DDDM-EVC\datasets\msp-podcast\Audio")
TARGET_PATH = Path(r"D:\DDDM-EVC\datasets\msp-podcast\Extracted")


if __name__ == "__main__":
    cfg = load_hydra_config("dddm_evc_xlsr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel_transform = MelTransform(cfg.data.mel_transform).to(device)

    (TARGET_PATH / "mel").mkdir(parents=True, exist_ok=True)
    (TARGET_PATH / "f0").mkdir(parents=True, exist_ok=True)

    filenames = list(AUDIO_PATH.glob("*.wav"))
    for i, fname in enumerate(filenames):
        audio, _ = torchaudio.load(AUDIO_PATH / fname)
        audio = audio.to(device)

        f0 = util.get_normalized_f0(audio, sr=16000)
        mel = mel_transform(audio)

        path = (AUDIO_PATH.parent / "Extracted").absolute()
        torch.save(mel, path / "mel" / f"{fname.stem}.pth")
        torch.save(f0, path / "f0" / f"{fname.stem}.pth")

        print(f"\rSaved sample {i} to {path}", end="")
