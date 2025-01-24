import torch

import util
from config import Config
from data import AudioDataloader, MelSpectrogramFixed
from models import SpeakerEncoder


def test_speaker_encoder(config: Config, dataloader: AudioDataloader) -> None:
    speaker_encoder = SpeakerEncoder(config.models.speaker_encoder)
    mel_transform = MelSpectrogramFixed(config.data.mel_transform)

    batch = next(iter(dataloader))

    x: torch.Tensor
    length: torch.Tensor
    x, length = batch

    x_mel = mel_transform(x)  # B x C x T
    mask = util.sequence_mask(length, x_mel.size(2)).to(x_mel.dtype)  # B x T
    mask = mask.unsqueeze(1)  # B x 1 x T

    output = speaker_encoder(x_mel, mask)
    assert output.shape[0] == config.training.batch_size
    assert output.shape[1] == config.models.speaker_encoder.out_dim
