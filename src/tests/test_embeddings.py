import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader, MelTransform
from models import (
    XLSR,
    Hubert,
    MetaStyleSpeech,
    StyleEncoder,
    VQVAEEncoder,
    W2V2LRobust,
)
from util import get_normalized_f0, load_model, sequence_mask


@pytest.mark.parametrize("config_name", ["dddm_vc_xlsr"])
def test_meta_style_speech(
    model_config: DictConfig, dataloader: AudioDataloader
) -> None:
    """Test Meta-StyleSpeech encoder."""
    mel_transform = MelTransform(model_config.data.mel_transform)
    speaker_encoder = MetaStyleSpeech(model_config.model.style_encoder)
    load_model(speaker_encoder, "metastylespeech.pth")

    x, x_n_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    x_mask = sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)  # B x T

    output = speaker_encoder(x, x_mel, x_mask)
    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.style_encoder.out_dim


@pytest.mark.parametrize("config_name", ["dddm_vc_xlsr"])
def test_vq_vae(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test VQ-VAE pitch encoder."""
    pitch_encoder = VQVAEEncoder(model_config.model.pitch_encoder)
    load_model(pitch_encoder, "vqvae.pth")

    x, _ = next(iter(dataloader))
    f0 = get_normalized_f0(x, model_config.data.mel_transform.sample_rate)

    output = pitch_encoder(f0)
    assert output.shape[0] == model_config.training.batch_size
    assert output.min().item() == 0
    assert output.max().item() == model_config.model.pitch_encoder.vq.k_bins - 1
    assert not output.is_floating_point()  # Discrete codes


@pytest.mark.parametrize("config_name", ["dddm_vc_xlsr"])
def test_xlsr(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    xlsr = XLSR()

    x, _ = next(iter(dataloader))

    output = xlsr(x)

    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.content_encoder.out_dim


@pytest.mark.parametrize("config_name", ["dddm_evc_hu"])
def test_hubert(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    hubert = Hubert()

    x, _ = next(iter(dataloader))

    output = hubert(x)

    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.content_encoder.out_dim


@pytest.mark.parametrize("config_name", ["dddm_vc_xlsr"])
def test_w2v2_l_robust(cfg_evc: DictConfig, dataloader: AudioDataloader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME).to(device)

    x, _ = next(iter(dataloader))
    x = x.to(device)

    emb, logits = model(x)

    assert emb.shape[0] == logits.shape[0] == cfg_evc.training.batch_size
    assert emb.shape[1] == cfg_evc.model.style_encoder.emotion_encoder.out_dim
    assert logits.shape[1] == 3


@pytest.mark.parametrize("config_name", ["dddm_vc_xlsr"])
def test_style_encoder(cfg_evc: DictConfig, dataloader: AudioDataloader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(cfg_evc.data.mel_transform).to(device)
    model = StyleEncoder(cfg_evc.model.style_encoder).to(device)

    x, x_n_frames = next(iter(dataloader))
    x = x.to(device)
    x_n_frames = x_n_frames.to(device)
    x_mel = mel_transform(x)
    x_mask = sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)  # B x T

    emb = model(x, x_mel, x_mask)

    cond_dim = (
        cfg_evc.model.style_encoder.emotion_emb_dim
        + cfg_evc.model.style_encoder.speaker_encoder.out_dim
    )
    assert emb.shape == (cfg_evc.training.batch_size, cond_dim)
