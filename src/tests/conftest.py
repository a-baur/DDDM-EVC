import pytest

from config import Config
from data import AudioDataloader, AudioDataset, load_librispeech

CONFIG_NAME = "config.yaml"


@pytest.fixture()  # type: ignore
def config() -> Config:
    return Config.from_yaml(CONFIG_NAME)


@pytest.fixture()  # type: ignore
def dataloader(config: Config) -> AudioDataloader:
    librispeech = load_librispeech(
        root=config.data.dataset.path, url="dev-clean", folder_in_archive="LibriSpeech"
    )
    dataset = AudioDataset(cfg=config, dataset=librispeech)
    dataloader = AudioDataloader(dataset, config)
    return dataloader
