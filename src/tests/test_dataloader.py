import pytest
from omegaconf import DictConfig

from data import MelTransform
from data.dataloader import AudioDataloader


@pytest.mark.parametrize(
    "config_name", ["dddm_vc_xlsr", "dddm_evc_xlsr", "dddm_evc_hu"]
)
def test_dataloader(dataloader: AudioDataloader, model_config: DictConfig) -> None:
    mel_transform = MelTransform(model_config.data.mel_transform)
    waveforms, lengths = next(iter(dataloader))
    mel = mel_transform(waveforms)

    assert waveforms is not None
    assert lengths is not None
    assert mel is not None

    n_frames = (
        model_config.data.dataset.segment_size
        // model_config.data.mel_transform.hop_length
    )
    assert mel.shape == (
        model_config.training.batch_size,
        model_config.data.mel_transform.n_mel_channels,
        n_frames,
    )
    assert waveforms.shape == (
        model_config.training.batch_size,
        model_config.data.dataset.segment_size,
    )
    assert lengths.shape == (model_config.training.batch_size,)
