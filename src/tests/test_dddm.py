import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader, MelTransform
from models import dddm_from_config
from util.helpers import move_to_device


@pytest.mark.parametrize(
    "config_name",
    ["dddm_vc_xlsr"],  # , "dddm_evc_xlsr", "dddm_evc_hu"]
)
def test_dddm(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test DDDM model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(model_config.data.mel_transform)
    model = dddm_from_config(model_config.model, pretrained=False)

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)

    y_mel, src_out, ftr_out = model(x, x_mel, x_n_frames, return_enc_out=True)
    assert y_mel.shape == x_mel.shape
    assert src_out.shape == x_mel.shape
    assert ftr_out.shape == x_mel.shape


@pytest.mark.parametrize(
    "config_name", ["dddm_vc_xlsr", "dddm_evc_xlsr", "dddm_evc_hu"]
)
def test_dddm_vc(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test DDDM voice conversion"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(model_config.data.mel_transform)
    model = dddm_from_config(model_config.model, pretrained=True)

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    t, t_frames = next(iter(dataloader))
    t_mel = mel_transform(t)

    x, x_mel, x_n_frames, t, t_mel, t_frames, model = move_to_device(
        (x, x_mel, x_n_frames, t, t_mel, t_frames, model), device
    )

    y_mel, src_out, ftr_out = model.voice_conversion(
        x, x_mel, x_n_frames, t, t_mel, t_frames, return_enc_out=True
    )
    assert y_mel.shape == x_mel.shape
    assert src_out.shape == x_mel.shape
    assert ftr_out.shape == x_mel.shape


@pytest.mark.parametrize(
    "config_name", ["dddm_vc_xlsr", "dddm_evc_xlsr", "dddm_evc_hu"]
)
def test_dddm_loss(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test DDDM model loss computation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(model_config.data.mel_transform)
    model = dddm_from_config(model_config.model, pretrained=True)

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)

    diff_loss, rec_loss = model.compute_loss(x, x_mel, x_n_frames)
    assert diff_loss >= 0
    assert rec_loss >= 0
