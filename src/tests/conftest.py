import pytest
from hydra import compose, initialize_config_dir

import config
from data import AudioDataloader, Librispeech, MSPPodcast, librispeech_collate_fn

CONFIG_NAME = "vc_config.yaml"
TESTING_DATASET = "librispeech"
# TESTING_DATASET = "msp-podcast"


@pytest.fixture()  # type: ignore
def cfg() -> config.Config:
    config.register_configs()
    with initialize_config_dir(
        version_base=None, config_dir=config.CONFIG_PATH.as_posix()
    ):
        cfg = compose(config_name="config_vc.yaml")
    return cfg._to_object()  # noqa


@pytest.fixture()  # type: ignore
def dataloader(cfg: config.Config) -> AudioDataloader:
    if TESTING_DATASET == "librispeech":
        dataset = Librispeech("dev-clean", "LibriSpeech")
        return AudioDataloader(
            dataset=dataset, cfg=cfg, collate_fn=librispeech_collate_fn
        )
    elif TESTING_DATASET == "msp-podcast":
        dataset = MSPPodcast(cfg.data, split="development")
        return AudioDataloader(dataset=dataset, cfg=cfg)
    else:
        raise ValueError(f"Unknown dataset: {TESTING_DATASET}")
