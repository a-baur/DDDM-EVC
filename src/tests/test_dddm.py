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
    # "evc_xlsr_yin",
    "evc_xlsr_yin_label",
    # "vc_xlsr_yin_dc",
]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_dddm_vc(
    model_config: DictConfig,
    device: torch.device,
    dataloader: AudioDataloader,
) -> None:
    """Test DDDM voice conversion"""
    model, preprocessor, style_encoder = models_from_config(model_config, device)

    audio, n_frames, x_labels = next(iter(dataloader))
    audio, n_frames, x_labels = (
        audio.to(device),
        n_frames.to(device),
        x_labels.to(device),
    )
    x = preprocessor(audio, n_frames, x_labels)

    audio, n_frames, t_labels = next(iter(dataloader))
    audio, n_frames, t_labels = (
        audio.to(device),
        n_frames.to(device),
        t_labels.to(device),
    )
    t = preprocessor(audio, n_frames, t_labels)

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

    audio, n_frames, labels = next(iter(dataloader))
    audio, n_frames, labels = audio.to(device), n_frames.to(device), labels.to(device)
    x = preprocessor(audio, n_frames, labels)
    g = style_encoder(x).unsqueeze(-1)

    if model_config.model.use_duration_control:
        dc = DurationControl(
            model_config.model.content_encoder.out_dim,
            model_config.model.style_encoder.out_dim,
        ).to(device)
        x, dur_loss = dc(x, g, return_loss=True)
        assert dur_loss >= 0

    score_loss, src_ftr_loss, rec_loss = model.compute_loss(x, g, rec_loss=False)
    print(score_loss, src_ftr_loss, rec_loss)
    assert score_loss >= 0
    assert src_ftr_loss >= 0
    assert rec_loss >= 0
