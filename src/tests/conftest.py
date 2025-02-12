import pytest

import config
from data import AudioDataloader, Librispeech, MSPPodcast, librispeech_collate_fn
from util import get_root_path

CONFIG_NAME = "config_vc.yaml"
TESTING_DATASET = "msp-podcast"  # "librispeech"


@pytest.fixture(autouse=True)  # type: ignore
def change_to_root_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(get_root_path())


@pytest.fixture()  # type: ignore
def cfg() -> config.Config:
    config.register_configs()
    return config.load_hydra_config(CONFIG_NAME)


@pytest.fixture()  # type: ignore
def dataloader(cfg: config.Config) -> AudioDataloader:
    if TESTING_DATASET == "librispeech":
        dataset = Librispeech("dev-clean", "LibriSpeech")
        return AudioDataloader(
            dataset=dataset, cfg=cfg, collate_fn=librispeech_collate_fn
        )
    elif TESTING_DATASET == "msp-podcast":
        dataset = MSPPodcast(cfg.data, split="development")
        return AudioDataloader(
            dataset=dataset, cfg=cfg, collate_fn=librispeech_collate_fn
        )
    else:
        raise ValueError(f"Unknown dataset: {TESTING_DATASET}")
