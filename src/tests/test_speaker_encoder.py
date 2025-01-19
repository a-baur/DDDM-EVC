from config import Config
from data.dataloader import AudioDataloader
from models import SpeakerEncoder


def test_speaker_encoder(config: Config, dataloader: AudioDataloader) -> None:
    speaker_encoder = SpeakerEncoder(config.models.speaker_encoder)
    print(speaker_encoder)
