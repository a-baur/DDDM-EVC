from config import Config
from data import MelTransform
from data.dataloader import AudioDataloader


def test_dataloader(dataloader: AudioDataloader, cfg: Config) -> None:
    mel_transform = MelTransform(cfg.data.mel_transform)
    waveforms, lengths = next(iter(dataloader))
    mel = mel_transform(waveforms)

    assert waveforms is not None
    assert lengths is not None
    assert mel is not None

    n_frames = cfg.data.dataset.segment_size // cfg.data.mel_transform.hop_length
    assert mel.shape == (
        cfg.training.batch_size,
        cfg.data.mel_transform.n_mel_channels,
        n_frames,
    )
    assert waveforms.shape == (cfg.training.batch_size, cfg.data.dataset.segment_size)
    assert lengths.shape == (cfg.training.batch_size,)
