from config import Config
from data import AudioDataloader, MelSpectrogramFixed
from models import VQVAE, SourceFilterEncoder, Wav2Vec2
from util import get_normalized_f0


def test_src_ftr_encoder(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Source-Filter encoder."""
    src_ftr_encoder = SourceFilterEncoder(cfg.models.src_ftr_encoder)
    content_encoder = Wav2Vec2()
    pitch_encoder = VQVAE(cfg.models.src_ftr_encoder.pitch_encoder)

    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)

    x, length = next(iter(dataloader))
    f0 = get_normalized_f0(x, cfg.data.mel_transform.sample_rate)

    x_mel = mel_transform(x)
    content_enc = content_encoder(x)
    pitch_enc = pitch_encoder.code_extraction(f0)

    speaker_enc, src_mel, ftr_mel = src_ftr_encoder(
        x_mel, length, content_enc, pitch_enc, mixup=False
    )

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape
    assert speaker_enc.shape == (
        cfg.training.batch_size,
        cfg.models.src_ftr_encoder.speaker_encoder.out_dim,
        1,
    )
