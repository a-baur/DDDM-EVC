import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader
from models import models_from_config
from models.dddm.duration_control import DurationControl

CONFIG_NAMES = [
    # "evc_xlsr",
    # "evc_hu",
    # "vc_xlsr_ph",
    # "vc_xlsr",
    # "evc_xlsr_ph",
    # "vc_xlsr_ph_yin",
    # "evc_xlsr_ph_yin",
    # "vc_xlsr_yin",
    "vc_xlsr_yin_dc",
]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_dddm_vc(
    model_config: DictConfig,
    device: torch.device,
    dataloader: AudioDataloader,
) -> None:
    """Test DDDM voice conversion"""
    model, preprocessor, style_encoder = models_from_config(model_config, device)

    audio, n_frames = next(iter(dataloader))
    audio, n_frames = audio.to(device), n_frames.to(device)
    x = preprocessor(audio)

    audio, n_frames = next(iter(dataloader))
    audio, n_frames = audio.to(device), n_frames.to(device)
    t = preprocessor(audio)

    g = style_encoder(t).unsqueeze(-1)

    if model_config.model.use_duration_control:
        dc = DurationControl(
            model_config.model.content_encoder.out_dim,
            model_config.model.style_encoder.out_dim,
        ).to(device)
        x = dc(x, g)

    y_mel, src_out, ftr_out = model(x, g, return_enc_out=True)
    assert y_mel.shape == x.mel.shape
    assert src_out.shape == x.mel.shape
    assert ftr_out.shape == x.mel.shape


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_dddm_loss(
    model_config: DictConfig,
    device: torch.device,
    dataloader: AudioDataloader,
) -> None:
    """Test DDDM model loss computation"""
    model, preprocessor, style_encoder = models_from_config(model_config, device)

    audio, n_frames = next(iter(dataloader))
    audio, n_frames = audio.to(device), n_frames.to(device)
    x = preprocessor(audio)
    g = style_encoder(x).unsqueeze(-1)

    if model_config.model.use_duration_control:
        dc = DurationControl(
            model_config.model.content_encoder.out_dim,
            model_config.model.style_encoder.out_dim,
        ).to(device)
        x, dur_loss = dc(x, g, return_loss=True)
        assert dur_loss >= 0

    diff_loss, rec_loss = model.compute_loss(x, g)
    assert diff_loss >= 0
    assert rec_loss >= 0
