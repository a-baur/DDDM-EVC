import os

import pytest

from config import Config
from data.datasets import MSPPodcast


def test_msp_podcast(cfg: Config) -> None:
    if not os.path.exists(cfg.data.dataset.path):
        pytest.skip()
    dataset = MSPPodcast(cfg.data, split="development")
    audio, length = dataset[0]
    assert audio.shape == (cfg.data.dataset.segment_size,)
