import torch

from config import Config
from data import AudioDataloader, MelTransform
from models import XLSR, EmotionModel, MetaStyleSpeech, VQVAEEncoder
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


def test_xlsr(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    xlsr = XLSR()

    x, _ = next(iter(dataloader))

    output = xlsr(x)

    assert output.shape[0] == cfg.training.batch_size


def test_w2v2_l_robust(cfg: Config, dataloader: AudioDataloader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmotionModel.from_pretrained(EmotionModel.MODEL_NAME)
    model.to(device)

    x_batch, _ = next(iter(dataloader))
    x_batch = x_batch.to(device)

    emb_batch, logits_batch = model(x_batch)

    x = x_batch[0]
    emb, logits = model(x)

    assert emb == emb_batch[0]
    assert logits == logits_batch[0]
