import torch

from config import Config
from data import AudioDataloader
from models import DDDM
from util.helpers import move_to_device


def test_dddm(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(freeze=True)

    x, x_mel, x_n_frames = next(iter(dataloader))

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)

    y_mel, enc_out = model(x, x_mel, x_n_frames, return_enc_out=True)
    assert y_mel.shape == x_mel.shape
    assert enc_out.shape == x_mel.shape


def test_dddm_vc(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM voice conversion"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(freeze=True)

    x, x_mel, x_n_frames = next(iter(dataloader))
    _y, _y_mel, _y_frames = next(iter(dataloader))

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)
    _y_mel, _y_frames = move_to_device((_y_mel, _y_frames), device)

    y_mel, enc_out = model.voice_conversion(
        x, x_mel, x_n_frames, _y_mel, _y_frames, return_enc_out=True
    )
    assert y_mel.shape == x_mel.shape
    assert enc_out.shape == x_mel.shape


def test_dddm_loss(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM model loss computation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(freeze=True)

    x, x_mel, x_n_frames = next(iter(dataloader))

    x, x_mel, x_n_frames, model = move_to_device((x, x_mel, x_n_frames, model), device)

    diff_loss, rec_loss = model.compute_loss(x, x_mel, x_n_frames)
    assert diff_loss >= 0
    assert rec_loss >= 0
