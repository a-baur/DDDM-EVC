from config import Config
from data.dataloader import AudioDataloader


def test_dataloader(dataloader: AudioDataloader, cfg: Config) -> None:
    waveforms, lengths = next(iter(dataloader))
    assert waveforms is not None
    assert lengths is not None
    assert len(waveforms) == cfg.training.batch_size
    assert len(lengths) == cfg.training.batch_size
