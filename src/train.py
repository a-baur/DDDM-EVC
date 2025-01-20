import torch

import util
from config import Config
from data import AudioDataloader, MelSpectrogramFixed, load_librispeech
from models import SpeakerEncoder

if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")

    dataset = load_librispeech(
        root=cfg.data.dataset.path, url="dev-clean", folder_in_archive="LibriSpeech"
    )
    dataloader = AudioDataloader(dataset, cfg)
    speaker_encoder = SpeakerEncoder(cfg.models.speaker_encoder)
    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)

    for batch in dataloader:
        x: torch.Tensor
        length: torch.Tensor
        x, length = batch

        x_mel = mel_transform(x)

        mask = torch.unsqueeze(util.sequence_mask(length, x_mel.size(2)), 1).to(
            x_mel.dtype
        )
        output = speaker_encoder(x, mask)
        print(output.shape)
        break
