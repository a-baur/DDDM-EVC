import torch

from config import Config
from data import AudioDataloader, MelTransform
from models import DDDM
from util.helpers import move_to_device


def test_dddm(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(cfg.data.mel_transform)
    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(mode="eval")

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)

    y_mel, src_out, ftr_out = model(x, x_mel, x_n_frames, return_enc_out=True)
    assert y_mel.shape == x_mel.shape
    assert src_out.shape == x_mel.shape
    assert ftr_out.shape == x_mel.shape


def test_dddm_vc(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM voice conversion"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(cfg.data.mel_transform)
    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(mode="eval")

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    t, t_frames = next(iter(dataloader))
    t_mel = mel_transform(t)

    x, x_mel, x_n_frames, t_mel, t_frames, model = move_to_device(
        (x, x_mel, x_n_frames, t_mel, t_frames, model), device
    )

    y_mel, src_out, ftr_out = model.voice_conversion(
        x, x_mel, x_n_frames, t_mel, t_frames, return_enc_out=True
    )
    assert y_mel.shape == x_mel.shape
    assert src_out.shape == x_mel.shape
    assert ftr_out.shape == x_mel.shape


def test_dddm_loss(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM model loss computation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(cfg.data.mel_transform)
    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(mode="eval")

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)

    diff_loss, rec_loss = model.compute_loss(x, x_mel, x_n_frames)
    assert diff_loss >= 0
    assert rec_loss >= 0
