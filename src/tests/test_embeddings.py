from config import Config
from data import AudioDataloader, MelTransform
from models import MetaStyleSpeech, VQVAEEncoder, Wav2Vec2
from util import get_normalized_f0, load_model, sequence_mask


def test_meta_style_speech(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Meta-StyleSpeech encoder."""
    mel_transform = MelTransform(cfg.data.mel_transform)
    speaker_encoder = MetaStyleSpeech(cfg.model.speaker_encoder)
    load_model(speaker_encoder, "metastylespeech.pth")

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    mask = sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)  # B x T

    output = speaker_encoder(x_mel, mask)
    assert output.shape[0] == cfg.training.batch_size
    assert output.shape[1] == cfg.model.speaker_encoder.out_dim


def test_vq_vae(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test VQ-VAE pitch encoder."""
    pitch_encoder = VQVAEEncoder(cfg.model.pitch_encoder)
    load_model(pitch_encoder, "vqvae.pth")

    x, _ = next(iter(dataloader))
    f0 = get_normalized_f0(x, cfg.data.mel_transform.sample_rate)

    output = pitch_encoder(f0)
    assert output.shape[0] == cfg.training.batch_size
    assert output.min().item() == 0
    assert output.max().item() == cfg.model.pitch_encoder.vq.k_bins - 1
    assert not output.is_floating_point()  # Discrete codes


def test_wav2vec2(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    wav2vec2 = Wav2Vec2()

    x, _ = next(iter(dataloader))

    output = wav2vec2(x)

    assert output.shape[0] == cfg.training.batch_size
