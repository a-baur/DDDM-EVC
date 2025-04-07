import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader
from models import XLSR, Hubert, VQVAEEncoder, models_from_config
from models.content_encoder import XLSR_ESPEAK_CTC
from models.pitch_encoder import YINEncoder
from util import get_normalized_f0, load_model


@pytest.mark.parametrize("config_name", ["vc_xlsr", "evc_xlsr", "evc_hu", "vc_xlsr_ph"])
def test_style_encoder(
    model_config: DictConfig, dataloader: AudioDataloader, device: torch.device
) -> None:
    """Test Meta-StyleSpeech encoder."""
    _, preprocessor, style_encoder = models_from_config(model_config, device)
    audio, n_frames, labels = next(iter(dataloader))
    audio, n_frames, labels = audio.to(device), n_frames.to(device), labels.to(device)
    x = preprocessor(audio)
    output = style_encoder(x)
    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.decoder.gin_channels
    assert output.shape[1] == model_config.model.diffusion.gin_channels


@pytest.mark.parametrize("config_name", ["vc_xlsr", "evc_xlsr", "evc_hu", "vc_xlsr_ph"])
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


@pytest.mark.parametrize("config_name", ["vc_xlsr", "evc_xlsr"])
def test_xlsr(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    xlsr = XLSR()
    x, _ = next(iter(dataloader))
    output = xlsr(x)

    expected_frames = x.shape[1] // model_config.data.mel_transform.hop_length
    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.content_encoder.out_dim
    assert output.shape[2] == expected_frames


@pytest.mark.parametrize("config_name", ["vc_xlsr_yin_dc"])
def test_xlsr_espeak_ctc(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    xlsr = XLSR_ESPEAK_CTC(return_logits=True, return_hidden=True)
    x, _ = next(iter(dataloader))
    logits, emb = xlsr(x)
    expected_frames = x.shape[1] // model_config.data.mel_transform.hop_length
    assert logits.shape[0] == model_config.training.batch_size
    assert logits.shape[1] == expected_frames
    assert emb.shape[0] == model_config.training.batch_size
    assert emb.shape[1] == model_config.model.content_encoder.out_dim
    assert emb.shape[2] == expected_frames


@pytest.mark.parametrize("config_name", ["evc_hu"])
def test_hubert(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    hubert = Hubert()
    x, _ = next(iter(dataloader))
    output = hubert(x)

    expected_frames = x.shape[1] // model_config.data.mel_transform.hop_length
    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.content_encoder.out_dim
    assert output.shape[2] == expected_frames


@pytest.mark.parametrize("config_name", ["evc_xlsr_yin"])
def test_yin_encoder(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test YIN pitch encoder."""
    yin = YINEncoder(model_config.pitch_encoder)
    x, _ = next(iter(dataloader))
    out = yin(x)
    print(out.shape)
    assert out.shape[0] == x.shape[0]
