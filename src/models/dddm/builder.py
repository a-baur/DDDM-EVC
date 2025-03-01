import typing

import torch
from omegaconf import DictConfig, OmegaConf

import util
from config import DDDM_EVC_HUBERT_Config, DDDM_EVC_XLSR_Config, DDDM_VC_XLSR_Config
from data import MelTransform
from models.content_encoder import XLSR, Hubert
from models.diffusion import Diffusion
from models.pitch_encoder import VQVAEEncoder
from models.style_encoder import MetaStyleSpeech, StyleEncoder
from modules.wavenet_decoder import WavenetDecoder

from .dddm import DDDM
from .input import DDDMPreprocessor

MODEL_BLUEPRINT = {
    DDDM_VC_XLSR_Config: {
        "style_encoder": MetaStyleSpeech,
        "content_encoder": XLSR,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    DDDM_EVC_XLSR_Config: {
        "style_encoder": StyleEncoder,
        "content_encoder": XLSR,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    DDDM_EVC_HUBERT_Config: {
        "style_encoder": StyleEncoder,
        "content_encoder": Hubert,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
}

MODEL_PATHS = {
    DDDM_VC_XLSR_Config: {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
        WavenetDecoder: "vc/wavenet_decoder.pth",
        Diffusion: "vc/diffusion.pth",
    },
    DDDM_EVC_XLSR_Config: {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
    DDDM_EVC_HUBERT_Config: {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
}


@typing.no_type_check
def models_from_config(
    cfg: DictConfig, device: torch.device = torch.device("cpu"), sample_rate: int = None
) -> tuple[DDDM, DDDMPreprocessor, StyleEncoder | MetaStyleSpeech]:
    """
    Builds DDDM model from configuration.

    :param cfg: ConfigVC or ConfigEVC
    :param device: Device to move model to
    :param sample_rate: Sample rate of data
    :return: DDDM model
    """
    if sample_rate is None:
        sample_rate = cfg.data.dataset.sampling_rate

    cfg_type = type(OmegaConf.to_object(cfg.model))
    if cfg_type not in MODEL_BLUEPRINT:
        raise ValueError(f"Unknown config type: {cfg_type.__name__}")

    # Retrieve model components and paths
    components = MODEL_BLUEPRINT[cfg_type]
    paths = MODEL_PATHS.get(cfg_type, {})

    ## Initialize style encoder
    style_encoder_cls = components["style_encoder"]
    style_encoder = style_encoder_cls(cfg.model.style_encoder).to(device)

    # Load style encoder model weights
    style_encoder_path = paths.get(style_encoder_cls)
    if style_encoder_path:
        if style_encoder_cls == MetaStyleSpeech:
            util.load_model(style_encoder, style_encoder_path, freeze=True)
        else:
            util.load_model(
                style_encoder.speaker_encoder,
                paths.get(MetaStyleSpeech, ""),
                freeze=True,
            )

    ## Initialize content and pitch encoders
    content_encoder_cls = components["content_encoder"]
    content_encoder = content_encoder_cls().to(device)
    if path := paths.get(content_encoder_cls):
        util.load_model(content_encoder, path, freeze=True)

    pitch_encoder_cls = components["pitch_encoder"]
    pitch_encoder = pitch_encoder_cls(cfg.model.pitch_encoder).to(device)
    if path := paths.get(pitch_encoder_cls):
        util.load_model(pitch_encoder, path, freeze=True)

    ## Initialize mel transform and preprocessor
    mel_transform = MelTransform(cfg.data.mel_transform)
    preprocessor = DDDMPreprocessor(
        mel_transform, pitch_encoder, content_encoder, sample_rate
    ).to(device)

    ## Initialize decoder and diffusion model
    decoder_cls = components["decoder"]
    decoder = decoder_cls(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        f0_dim=cfg.model.pitch_encoder.vq.k_bins,
    ).to(device)
    if path := paths.get(decoder_cls):
        util.load_model(decoder, path)

    diffusion_cls = components["diffusion"]
    diffusion = diffusion_cls(cfg.model.diffusion).to(device)
    if path := paths.get(diffusion_cls):
        util.load_model(diffusion, path)

    ## Finalize and return model components
    model = DDDM(decoder, diffusion).to(device)
    return model, preprocessor, style_encoder
