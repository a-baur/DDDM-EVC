from config import Config
from data.dataloader import AudioDataloader


def test_dataloader(dataloader: AudioDataloader, config: Config) -> None:
    waveforms, sample_rates = next(iter(dataloader))
    assert waveforms is not None
    assert sample_rates is not None
    assert len(waveforms) == config.training.batch_size
    assert len(sample_rates) == config.training.batch_size
