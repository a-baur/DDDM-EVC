import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader
from models import models_from_config

CONFIG_NAMES = [
    # "dddm_evc_xlsr",
    # "dddm_evc_hu",
    # "dddm_vc_xlsr_ph",
    # "dddm_vc_xlsr",
    # "dddm_evc_xlsr_ph",
    # "dddm_vc_xlsr_ph_yin",
    "dddm_evc_xlsr_ph_yin",
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
    diff_loss, rec_loss = model.compute_loss(x, g)
    assert diff_loss >= 0
    assert rec_loss >= 0
