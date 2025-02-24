import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader, MelTransform
from models import XLSR, Hubert, MetaStyleSpeech, StyleEncoder, VQVAEEncoder
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


@pytest.mark.parametrize("config_name", ["dddm_evc_xlsr", "dddm_evc_hu"])
def test_style_encoder(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mel_transform = MelTransform(model_config.data.mel_transform).to(device)
    model = StyleEncoder(model_config.model.style_encoder).to(device)

    x, x_n_frames = next(iter(dataloader))
    x = x.to(device)
    x_n_frames = x_n_frames.to(device)
    x_mel = mel_transform(x)
    x_mask = sequence_mask(x_n_frames, x_mel.size(2)).to(x_mel.dtype)  # B x T

    emb = model(x, x_mel, x_mask)

    cond_dim = (
        model_config.model.style_encoder.emotion_emb_dim
        + model_config.model.style_encoder.speaker_encoder.out_dim
    )
    assert emb.shape == (model_config.training.batch_size, cond_dim)


@pytest.mark.parametrize(
    "config_name", ["dddm_vc_xlsr", "dddm_evc_xlsr", "dddm_evc_hu"]
)
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

    expected_frames = x.shape[1] // model_config.data.mel_transform.hop_length
    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.content_encoder.out_dim
    assert output.shape[2] == expected_frames


@pytest.mark.parametrize("config_name", ["dddm_evc_hu"])
def test_hubert(model_config: DictConfig, dataloader: AudioDataloader) -> None:
    """Test Wav2Vec2 encoder."""
    hubert = Hubert()

    x, _ = next(iter(dataloader))

    output = hubert(x)

    expected_frames = x.shape[1] // model_config.data.mel_transform.hop_length
    assert output.shape[0] == model_config.training.batch_size
    assert output.shape[1] == model_config.model.content_encoder.out_dim
    assert output.shape[2] == expected_frames
