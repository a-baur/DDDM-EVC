import torch

from config import ConfigVC
from data import AudioDataloader, MelTransform
from models import XLSR, MetaStyleSpeech, VQVAEEncoder, W2V2LRobust
from util import get_normalized_f0, load_model, sequence_mask


def test_meta_style_speech(cfg_vc: ConfigVC, dataloader: AudioDataloader) -> None:
    """Test Meta-StyleSpeech encoder."""
    mel_transform = MelTransform(cfg_vc.data.mel_transform)
    speaker_encoder = MetaStyleSpeech(cfg_vc.model.speaker_encoder)
    load_model(speaker_encoder, "metastylespeech.pth")

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    mask = sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)  # B x T

    output = speaker_encoder(x_mel, mask)
    assert output.shape[0] == cfg_vc.training.batch_size
    assert output.shape[1] == cfg_vc.model.speaker_encoder.out_dim


def test_vq_vae(cfg_vc: ConfigVC, dataloader: AudioDataloader) -> None:
    """Test VQ-VAE pitch encoder."""
    pitch_encoder = VQVAEEncoder(cfg_vc.model.pitch_encoder)
    load_model(pitch_encoder, "vqvae.pth")

    x, _ = next(iter(dataloader))
    f0 = get_normalized_f0(x, cfg_vc.data.mel_transform.sample_rate)

    output = pitch_encoder(f0)
    assert output.shape[0] == cfg_vc.training.batch_size
    assert output.min().item() == 0
    assert output.max().item() == cfg_vc.model.pitch_encoder.vq.k_bins - 1
    assert not output.is_floating_point()  # Discrete codes


def test_xlsr(cfg_vc: ConfigVC, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    xlsr = XLSR()

    x, _ = next(iter(dataloader))

    output = xlsr(x)

    assert output.shape[0] == cfg_vc.training.batch_size
    assert output.shape[1] == 1024


def test_w2v2_l_robust(cfg_vc: ConfigVC, dataloader: AudioDataloader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME)
    model.to(device)

    x, _ = next(iter(dataloader))
    x = x.to(device)

    emb, logits = model(x)

    assert emb.shape[0] == logits.shape[0] == cfg_vc.training.batch_size
    assert emb.shape[1] == 1024
    assert logits.shape[1] == 3
