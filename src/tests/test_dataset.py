import os

import pytest
from omegaconf import DictConfig

from data.datasets import MSPPodcast


@pytest.mark.parametrize(
    "config_name", ["dddm_vc_xlsr", "dddm_evc_xlsr", "dddm_evc_hu"]
)
def test_msp_podcast(model_config: DictConfig) -> None:
    if not os.path.exists(model_config.data.dataset.path):
        pytest.skip()
    dataset = MSPPodcast(model_config.data, split="development")
    audio, length = dataset[0]
    assert audio.shape == (model_config.data.dataset.segment_size,)
