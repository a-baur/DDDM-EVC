import pytest

from config import Config
from data import AudioDataloader, MSPPodcast

CONFIG_NAME = "config.yaml"


@pytest.fixture()  # type: ignore
def config() -> Config:
    return Config.from_yaml(CONFIG_NAME)


@pytest.fixture()  # type: ignore
def dataloader(config: Config) -> AudioDataloader:
    dataset = MSPPodcast(config.data.dataset, split="development")
    dataloader = AudioDataloader(dataset=dataset, cfg=config)
    return dataloader
