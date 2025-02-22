import util
from config import ConfigVC
from data import AudioDataloader, MelTransform
from models import MetaStyleSpeech, SourceFilterEncoder


def test_source_filter_encoder(cfg_vc: ConfigVC, dataloader: AudioDataloader) -> None:
    """Test source-filter encoder."""
    mel_transfom = MelTransform(cfg_vc.data.mel_transform)
    src_ftr_encoder = SourceFilterEncoder(
        cfg_vc.model, sample_rate=cfg_vc.data.dataset.sampling_rate
    )
    speaker_encoder = MetaStyleSpeech(cfg_vc.model.speaker_encoder)

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transfom(x)
    x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)
    g = speaker_encoder(x, x_mel, x_mask).unsqueeze(-1)

    src_mel, ftr_mel = src_ftr_encoder(x, x_mask, g)

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape


def test_source_filter_voice_conversion(
    cfg_vc: ConfigVC, dataloader: AudioDataloader
) -> None:
    """Test source-filter encoder voice conversion."""
    mel_transfom = MelTransform(cfg_vc.data.mel_transform)
    src_ftr_encoder = SourceFilterEncoder(
        cfg_vc.model, sample_rate=cfg_vc.data.dataset.sampling_rate
    )
    speaker_encoder = MetaStyleSpeech(cfg_vc.model.speaker_encoder)

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transfom(x)
    x_mask = util.sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)

    y, y_frames = next(iter(dataloader))
    y_mel = mel_transfom(y)
    y_mask = util.sequence_mask(y_frames, y_mel.size(2)).to(y_mel.dtype)
    g = speaker_encoder(y, y_mel, y_mask).unsqueeze(-1)

    src_mel, ftr_mel = src_ftr_encoder(x, x_mask, g)

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape
