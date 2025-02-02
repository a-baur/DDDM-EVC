import util
from config import Config
from data import AudioDataloader, MelSpectrogramFixed
from models import VQVAE, MetaStyleSpeech, SourceFilterEncoder, Wav2Vec2
from util import get_normalized_f0


def test_src_ftr_encoder(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Source-Filter encoder."""
    # Load models
    content_encoder = Wav2Vec2()
    pitch_encoder = VQVAE(cfg.models.src_ftr_encoder.pitch_encoder)
    speaker_encoder = MetaStyleSpeech(cfg.models.src_ftr_encoder.speaker_encoder)
    src_ftr_encoder = SourceFilterEncoder(cfg.models.src_ftr_encoder)
    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)

    # Load data
    x, length = next(iter(dataloader))
    f0 = get_normalized_f0(x, cfg.data.mel_transform.sample_rate)
    x_mel = mel_transform(x)

    # Create mask
    x_mask = util.sequence_mask(length, x_mel.size(2)).to(x_mel.dtype)

    # Embeddings
    x_emb_content = content_encoder(x)
    x_emb_pitch = pitch_encoder.code_extraction(f0)
    x_emb_spk = speaker_encoder(x_mel, x_mask).unsqueeze(-1)

    # Encode embeddings into source and filter representations
    src_mel, ftr_mel = src_ftr_encoder(
        x_emb_spk, x_emb_content, x_emb_pitch, x_mask, mixup=False
    )

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape


def test_src_ftr_vc(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Source-Filter encoder."""
    content_encoder = Wav2Vec2()
    pitch_encoder = VQVAE(cfg.models.src_ftr_encoder.pitch_encoder)
    speaker_encoder = MetaStyleSpeech(cfg.models.src_ftr_encoder.speaker_encoder)
    src_ftr_encoder = SourceFilterEncoder(cfg.models.src_ftr_encoder)
    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)

    x, x_length = next(iter(dataloader))
    f0 = get_normalized_f0(x, cfg.data.mel_transform.sample_rate)
    x_mel = mel_transform(x)

    x_mask = util.sequence_mask(x_length, x_mel.size(2)).to(x_mel.dtype)

    y, y_length = next(iter(dataloader))
    y_mel = mel_transform(y)

    y_mask = util.sequence_mask(y_length, y_mel.size(2)).to(y_mel.dtype)

    x_emb_content = content_encoder(x)
    x_emb_pitch = pitch_encoder.code_extraction(f0)
    y_emb_spk = speaker_encoder(y_mel, y_mask).unsqueeze(-1)

    src_mel, ftr_mel = src_ftr_encoder(
        y_emb_spk, x_emb_content, x_emb_pitch, x_mask, mixup=False
    )

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape
