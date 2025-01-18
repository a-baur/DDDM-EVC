from config.schema import Config
from data.dataloader import AudioDataloader
from data.datasets import load_librispeech

if __name__ == "__main__":
    cfg = Config.from_yaml("config/config.yaml")

    dataset = load_librispeech(
        root=cfg.dataset.path, url="dev-clean", folder_in_archive="LibriSpeech"
    )

    dataloader = AudioDataloader(dataset, cfg.training.batch_size, cfg.dataloader)
    print(next(iter(dataloader)))
