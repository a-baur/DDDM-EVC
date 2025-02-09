import os

import pytest

from config import Config
from data.datasets import MSPPodcast

MSP_PODCAST_AVAILABLE = os.path.exists(
    Config.from_yaml("config.yaml").data.dataset.path
)


@pytest.mark.skipif(
    not MSP_PODCAST_AVAILABLE, reason="MSP Podcast dataset not found"
)  # type: ignore
def test_msp_podcast() -> None:
    cfg = Config.from_yaml("config.yaml")
    dataset = MSPPodcast(cfg.data, split="development")
    audio, length = dataset[0]
    assert audio.shape == (cfg.data.dataset.segment_size,)
