import pytest

import config
from data import AudioDataloader, Librispeech, MSPPodcast, librispeech_collate_fn
from util import get_root_path

TESTING_DATASET = "msp-podcast"  # "librispeech"


@pytest.fixture(autouse=True)  # type: ignore
def change_to_root_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(get_root_path())


@pytest.fixture()  # type: ignore
def cfg_vc() -> config.ConfigVC:
    config.register_configs()
    return config.load_hydra_config_vc("config_vc.yaml")


@pytest.fixture()  # type: ignore
def cfg_evc() -> config.ConfigEVC:
    config.register_configs()
    return config.load_hydra_config_evc("config_evc.yaml")


@pytest.fixture()  # type: ignore
def dataloader(cfg_vc: config.ConfigVC) -> AudioDataloader:
    if TESTING_DATASET == "librispeech":
        dataset = Librispeech("dev-clean", "LibriSpeech")
        return AudioDataloader(
            dataset=dataset,
            cfg=cfg_vc.data.dataloader,
            batch_size=cfg_vc.training.batch_size,
            collate_fn=librispeech_collate_fn,
        )
    elif TESTING_DATASET == "msp-podcast":
        dataset = MSPPodcast(cfg_vc.data, split="development")
        return AudioDataloader(
            dataset=dataset,
            cfg=cfg_vc.data.dataloader,
            batch_size=cfg_vc.training.batch_size,
            collate_fn=librispeech_collate_fn,
        )
    else:
        raise ValueError(f"Unknown dataset: {TESTING_DATASET}")
