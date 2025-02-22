import os

import pytest

from config import ConfigVC
from data.datasets import MSPPodcast


def test_msp_podcast(cfg_vc: ConfigVC) -> None:
    if not os.path.exists(cfg_vc.data.dataset.path):
        pytest.skip()
    dataset = MSPPodcast(cfg_vc.data, split="development")
    audio, length = dataset[0]
    assert audio.shape == (cfg_vc.data.dataset.segment_size,)
