from config import Config
from data import load_librispeech, AudioDataloader
from models import SpeakerEncoder


if __name__ == "__main__":
    cfg = Config.from_yaml("config.yaml")

    dataset = load_librispeech(
        root=cfg.dataset.path, url="dev-clean", folder_in_archive="LibriSpeech"
    )
    dataloader = AudioDataloader(dataset, cfg.training.batch_size, cfg.dataloader)

    speaker_encoder = SpeakerEncoder(cfg.models.speaker_encoder)
    print(speaker_encoder)
    print(next(iter(dataloader)))
