import util
from config import Config
from data import AudioDataloader, MelTransform
from models import MetaStyleSpeech, SourceFilterEncoder, VQVAEEncoder, WavenetDecoder
from util.helpers import load_model


def test_source_filter_encoder(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test source-filter encoder."""
    mel_transform = MelTransform(cfg.data.mel_transform)
    src_ftr_encoder = SourceFilterEncoder(
        cfg.model, sample_rate=cfg.data.dataset.sampling_rate
    )
    speaker_encoder = MetaStyleSpeech(cfg.model.speaker_encoder)

    x, x_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    x_mask = util.sequence_mask(x_frames, x_mel.size(2)).to(x_mel.dtype)
    g = speaker_encoder(x_mel, x_mask).unsqueeze(-1)

    src_mel, ftr_mel = src_ftr_encoder(x, x_mask, g)

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape


def test_source_filter_voice_conversion(
    cfg: Config, dataloader: AudioDataloader
) -> None:
    """Test source-filter encoder voice conversion."""
    mel_transform = MelTransform(cfg.data.mel_transform)
    src_ftr_encoder = SourceFilterEncoder(
        cfg.model, sample_rate=cfg.data.dataset.sampling_rate
    )
    speaker_encoder = MetaStyleSpeech(cfg.model.speaker_encoder)

    x, x_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    x_mask = util.sequence_mask(x_frames, x_mel.size(2)).to(x_mel.dtype)

    y, y_frames = next(iter(dataloader))
    y_mel = mel_transform(y)
    y_mask = util.sequence_mask(y_frames, y_mel.size(2)).to(y_mel.dtype)
    g = speaker_encoder(y_mel, y_mask).unsqueeze(-1)

    src_mel, ftr_mel = src_ftr_encoder(x, x_mask, g)

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape


def test_from_pretrained(cfg: Config, dataloader: AudioDataloader) -> None:
    """Test source-filter encoder."""

    mel_transform = MelTransform(cfg.data.mel_transform)

    pitch_encoder = VQVAEEncoder(cfg.model.pitch_encoder)
    load_model(pitch_encoder, "vqvae.pth", freeze=True)
    speaker_encoder = MetaStyleSpeech(cfg.model.speaker_encoder)
    load_model(speaker_encoder, "metastylespeech.pth", freeze=True)
    decoder = WavenetDecoder(cfg.model)
    load_model(decoder, "wavenet_decoder.pth", freeze=True)

    src_ftr_encoder = SourceFilterEncoder(
        cfg.model,
        sample_rate=cfg.data.dataset.sampling_rate,
        pitch_encoder=pitch_encoder,
        decoder=decoder,
    )

    x, x_frames = next(iter(dataloader))
    x_mel = mel_transform(x)
    x_mask = util.sequence_mask(x_frames, x_mel.size(2)).to(x_mel.dtype)
    g = speaker_encoder(x_mel, x_mask).unsqueeze(-1)

    src_mel, ftr_mel = src_ftr_encoder(x, x_mask, g)

    # # pack samples and save
    # torch.save(
    #     {
    #         "x": x,
    #         "x_mel": x_mel,
    #         "x_frames": x_frames,
    #         "spk_emb": speaker_encoder(x_mel, x_mask),
    #         "src_mel": src_mel,
    #         "ftr_mel": ftr_mel,
    #     },
    #     "testdata.pth",
    # )

    assert src_mel.shape == x_mel.shape
    assert ftr_mel.shape == x_mel.shape
