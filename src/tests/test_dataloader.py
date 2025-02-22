from config import ConfigVC
from data import MelTransform
from data.dataloader import AudioDataloader


def test_dataloader(dataloader: AudioDataloader, cfg_vc: ConfigVC) -> None:
    mel_transform = MelTransform(cfg_vc.data.mel_transform)
    waveforms, lengths = next(iter(dataloader))
    mel = mel_transform(waveforms)

    assert waveforms is not None
    assert lengths is not None
    assert mel is not None

    n_frames = cfg_vc.data.dataset.segment_size // cfg_vc.data.mel_transform.hop_length
    assert mel.shape == (
        cfg_vc.training.batch_size,
        cfg_vc.data.mel_transform.n_mel_channels,
        n_frames,
    )
    assert waveforms.shape == (
        cfg_vc.training.batch_size,
        cfg_vc.data.dataset.segment_size,
    )
    assert lengths.shape == (cfg_vc.training.batch_size,)
