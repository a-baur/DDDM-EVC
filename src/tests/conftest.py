import pytest
import torch
from omegaconf import DictConfig

import config
from data import AudioDataloader, Librispeech, MSPPodcast, librispeech_collate_fn
from models import DDDMInput, models_from_config
from util import get_root_path

TESTING_DATASET = "msp-podcast"  # "librispeech"


@pytest.fixture(autouse=True)  # type: ignore
def change_to_root_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(get_root_path())


@pytest.fixture  # type: ignore
def model_config(config_name: str) -> DictConfig:
    return config.load_hydra_config(config_name)


@pytest.fixture  # type: ignore
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture  # type: ignore
def dddm_input(
    model_config: DictConfig, dataloader: AudioDataloader, device: torch.device
) -> DDDMInput:
    _, preprocessor, _ = models_from_config(model_config, device)
    x, x_n_frames, labels = next(iter(dataloader))
    x = preprocessor(x, x_n_frames, labels)
    return x


@pytest.fixture  # type: ignore
def dataloader(model_config: DictConfig) -> AudioDataloader:
    if TESTING_DATASET == "librispeech":
        dataset = Librispeech("dev-clean", "LibriSpeech")
        return AudioDataloader(
            dataset=dataset,
            cfg=model_config.data.dataloader,
            batch_size=model_config.training.batch_size,
            collate_fn=librispeech_collate_fn,
        )
    elif TESTING_DATASET == "msp-podcast":
        dataset = MSPPodcast(model_config.data, split="development")
        return AudioDataloader(
            dataset=dataset,
            cfg=model_config.data.dataloader,
            batch_size=model_config.training.batch_size,
            collate_fn=librispeech_collate_fn,
        )
    else:
        raise ValueError(f"Unknown dataset: {TESTING_DATASET}")
