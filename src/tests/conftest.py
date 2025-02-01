import pytest

from config import Config
from data import AudioDataloader, Librispeech, MSPPodcast, librispeech_collate_fn

CONFIG_NAME = "config.yaml"
TESTING_DATASET = "librispeech"
# TESTING_DATASET = "msp-podcast"


@pytest.fixture()  # type: ignore
def cfg() -> Config:
    return Config.from_yaml(CONFIG_NAME)


@pytest.fixture()  # type: ignore
def dataloader(cfg: Config) -> AudioDataloader:
    if TESTING_DATASET == "librispeech":
        dataset = Librispeech("dev-clean", "LibriSpeech")
        return AudioDataloader(
            dataset=dataset, cfg=cfg, collate_fn=librispeech_collate_fn
        )
    elif TESTING_DATASET == "msp-podcast":
        dataset = MSPPodcast(cfg.data.dataset, split="development")
        return AudioDataloader(dataset=dataset, cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset: {TESTING_DATASET}")
