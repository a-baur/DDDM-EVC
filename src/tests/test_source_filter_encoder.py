import util
from config import Config
from data import AudioDataloader, MelSpectrogramFixed
from models import SourceFilterEncoder


def test_source_filter_encoder(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test source-filter encoder."""
    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)
    src_ftr_encoder = SourceFilterEncoder(cfg.models.src_ftr_encoder)

    x, x_length = next(iter(dataloader))
    x_mel = mel_transform(x)
    x_mask = util.sequence_mask(x_length, x_mel.size(2)).to(x_mel.dtype)

    src_mel, ftr_mel = src_ftr_encoder(x, x_mel, x_mask)

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape


def test_source_filter_voice_conversion(
    cfg: Config, dataloader: AudioDataloader
) -> None:
    """Test source-filter encoder voice conversion."""
    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)
    src_ftr_encoder = SourceFilterEncoder(cfg.models.src_ftr_encoder)

    x, x_length = next(iter(dataloader))
    x_mel = mel_transform(x)
    x_mask = util.sequence_mask(x_length, x_mel.size(2)).to(x_mel.dtype)

    y, y_length = next(iter(dataloader))
    y_mel = mel_transform(y)
    y_mask = util.sequence_mask(y_length, y_mel.size(2)).to(y_mel.dtype)

    src_mel, ftr_mel = src_ftr_encoder.voice_conversion(x, x_mask, y_mel, y_mask)

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape
