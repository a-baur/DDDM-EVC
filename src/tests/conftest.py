import pytest

from config import Config
from data.dataloader import AudioDataloader
from data.datasets import load_librispeech

CONFIG_NAME = "config.yaml"


@pytest.fixture()  # type: ignore
def config() -> Config:
    return Config.from_yaml(CONFIG_NAME)


@pytest.fixture()  # type: ignore
def dataloader(config: Config) -> AudioDataloader:
    dataset = load_librispeech(
        root=config.dataset.path, url="dev-clean", folder_in_archive="LibriSpeech"
    )
    dataloader = AudioDataloader(dataset, config.training.batch_size, config.dataloader)
    return dataloader
