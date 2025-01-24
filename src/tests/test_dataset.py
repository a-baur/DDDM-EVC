from data.datasets import MSPPodcast
from config import Config


def test_msp_podcast() -> None:
    cfg = Config.from_yaml("config.yaml")
    sr = cfg.data.dataset.sampling_rate
    dataset = MSPPodcast(cfg.data.dataset, split="development")
    audio, length = dataset[0]
    assert audio.shape[0] == 1
    assert audio.shape[1] == int(sr * length)
