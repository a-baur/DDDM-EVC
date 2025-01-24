from config import Config
from data.datasets import MSPPodcast


def test_msp_podcast() -> None:
    cfg = Config.from_yaml("config.yaml")
    dataset = MSPPodcast(cfg.data.dataset, split="development")
    audio, length = dataset[0]
    assert audio.shape == (cfg.data.dataset.segment_size,)
