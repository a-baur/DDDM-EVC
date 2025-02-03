import torch

from config import Config
from data import AudioDataloader, MelSpectrogramFixed
from models import VQVAE, MetaStyleSpeech, Wav2Vec2
from util import get_normalized_f0, get_root_path, sequence_mask


def test_meta_style_speech(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Meta-StyleSpeech encoder."""
    speaker_encoder = MetaStyleSpeech(cfg.models.src_ftr_encoder.speaker_encoder)

    ckpt_file = get_root_path() / "ckpt" / "metastylespeech.pth"
    speaker_encoder.load_state_dict(
        torch.load(ckpt_file, map_location="cpu", weights_only=True)
    )

    mel_transform = MelSpectrogramFixed(cfg.data.mel_transform)

    x, length = next(iter(dataloader))

    x_mel = mel_transform(x)  # B x C x T
    mask = sequence_mask(length, x_mel.size(2)).to(x_mel.dtype)  # B x T

    output = speaker_encoder(x_mel, mask)
    assert output.shape[0] == cfg.training.batch_size
    assert output.shape[1] == cfg.models.src_ftr_encoder.speaker_encoder.out_dim


def test_vq_vae(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test VQ-VAE pitch encoder."""
    pitch_encoder = VQVAE(cfg.models.src_ftr_encoder.pitch_encoder)

    ckpt_file = get_root_path() / "ckpt" / "vqvae.pth"
    pitch_encoder.load_state_dict(
        torch.load(ckpt_file, map_location="cpu", weights_only=True)
    )

    x, _ = next(iter(dataloader))
    f0 = get_normalized_f0(x, cfg.data.mel_transform.sample_rate)

    output = pitch_encoder.code_extraction(f0)
    assert output.shape[0] == cfg.training.batch_size
    assert output.min().item() == 0
    assert output.max().item() == cfg.models.src_ftr_encoder.pitch_encoder.vq.k_bins - 1
    assert not output.is_floating_point()  # Discrete codes


def test_wav2vec2(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    wav2vec2 = Wav2Vec2()

    x, _ = next(iter(dataloader))

    output = wav2vec2(x)

    assert output.shape[0] == cfg.training.batch_size
