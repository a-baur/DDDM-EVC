from pathlib import Path

import torch
import torchaudio

from config import load_hydra_config
from models import preprocessor_from_config, style_encoder_from_config

AUDIO_PATH = Path(r"D:\DDDM-EVC\datasets\msp-podcast\Audio")


if __name__ == "__main__":
    cfg = load_hydra_config("dddm_evc_xlsr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = preprocessor_from_config(cfg, device, sample_rate=16000)
    style_encoder = style_encoder_from_config(cfg, device)

    filenames = list(AUDIO_PATH.glob("*.wav"))

    for i, fname in enumerate(filenames):
        audio, _ = torchaudio.load(AUDIO_PATH / fname)
        audio = audio.to(device)

        x = preprocessor(audio)
        g = style_encoder(x)

        path = (AUDIO_PATH.parent / "Extracted").absolute()
        torch.save(x.mel, path / "mel" / f"{fname.stem}.pth")
        torch.save(x.emb_pitch, path / "emb_pitch" / f"{fname.stem}.pth")
        torch.save(x.emb_content, path / "emb_content" / f"{fname.stem}.pth")
        torch.save(g, path / "style" / f"{fname.stem}.pth")

        print(f"\rSaved sample {i} to {path}", end="")
