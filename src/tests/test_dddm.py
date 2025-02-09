from config import Config
from data import AudioDataloader, MelTransform
from models import DDDM


def test_dddm(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM model"""
    mel_transform = MelTransform(cfg.data.mel_transform)

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(freeze=True)

    x, x_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    y_mel, enc_out = model(x, x_mel, x_frames, return_enc_out=True)
    assert y_mel.shape == x_mel.shape
    assert enc_out.shape == x_mel.shape


def test_dddm_vc(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM voice conversion"""
    mel_transform = MelTransform(cfg.data.mel_transform)

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(freeze=True)

    x, x_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    _y, _y_frames = next(iter(dataloader))
    _y_mel = mel_transform(_y)

    y_mel, enc_out = model.voice_conversion(
        x, x_mel, x_frames, _y_mel, _y_frames, return_enc_out=True
    )
    assert y_mel.shape == x_mel.shape
    assert enc_out.shape == x_mel.shape


def test_dddm_loss(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test DDDM model loss computation"""
    mel_transform = MelTransform(cfg.data.mel_transform)

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(freeze=True)

    x, x_frames = next(iter(dataloader))
    x_mel = mel_transform(x)

    diff_loss, rec_loss = model.compute_loss(x, x_mel, x_frames)
    assert diff_loss >= 0
    assert rec_loss >= 0
