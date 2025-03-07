import torch
from omegaconf import DictConfig

import util
from data import MelTransform
from models.content_encoder import XLSR, XLSR_ESPEAK_CTC, Hubert
from models.diffusion import Diffusion
from models.pitch_encoder import VQVAEEncoder, YINEncoder
from models.style_encoder import MetaStyleSpeech, StyleEncoder
from modules.wavenet_decoder import WavenetDecoder

from .dddm import DDDM
from .input import DDDMPreprocessor

MODEL_BLUEPRINT = {
    "VC_XLSR": {
        "style_encoder": MetaStyleSpeech,
        "content_encoder": XLSR,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    "VC_XLSR_PH": {
        "style_encoder": MetaStyleSpeech,
        "content_encoder": XLSR_ESPEAK_CTC,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    "VC_XLSR_PH_YIN": {
        "style_encoder": MetaStyleSpeech,
        "content_encoder": XLSR_ESPEAK_CTC,
        "pitch_encoder": YINEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    "EVC_XLSR_PH_YIN": {
        "style_encoder": StyleEncoder,
        "content_encoder": XLSR_ESPEAK_CTC,
        "pitch_encoder": YINEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    "EVC_XLSR": {
        "style_encoder": StyleEncoder,
        "content_encoder": XLSR,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    "EVC_XLSR_PH": {
        "style_encoder": StyleEncoder,
        "content_encoder": XLSR_ESPEAK_CTC,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
    "EVC_HUBERT": {
        "style_encoder": StyleEncoder,
        "content_encoder": Hubert,
        "pitch_encoder": VQVAEEncoder,
        "decoder": WavenetDecoder,
        "diffusion": Diffusion,
    },
}

MODEL_PATHS = {
    "VC_XLSR": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
        WavenetDecoder: "vc/wavenet_decoder.pth",
        Diffusion: "vc/diffusion.pth",
    },
    "VC_XLSR_PH": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
    "VC_XLSR_PH_YIN": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
    "EVC_XLSR_PH_YIN": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
    "EVC_XLSR": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
    "EVC_XLSR_PH": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
    "EVC_HUBERT": {
        MetaStyleSpeech: "metastylespeech.pth",
        VQVAEEncoder: "vqvae.pth",
    },
}


def _comps_and_paths_from_config(
    cfg: DictConfig,
) -> tuple[dict[str, type], dict[type, str]]:
    component_id = cfg.component_id
    if component_id not in MODEL_BLUEPRINT:
        raise ValueError(f"Unknown component id: {component_id}")

    return MODEL_BLUEPRINT[component_id], MODEL_PATHS.get(component_id, {})


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
    style_encoder = style_encoder_from_config(cfg, device)
    preprocessor = preprocessor_from_config(cfg, device, sample_rate)
    model = dddm_from_config(cfg, device)

    return model, preprocessor, style_encoder


def preprocessor_from_config(
    cfg: DictConfig,
    device: torch.device,
    sample_rate: int | None,
) -> DDDMPreprocessor:
    components, paths = _comps_and_paths_from_config(cfg)
    if sample_rate is None:
        sample_rate = cfg.data.dataset.sampling_rate

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
        mel_transform,
        pitch_encoder,
        content_encoder,
        sample_rate,
        cfg.model.perturb_inputs,
    ).to(device)

    return preprocessor


def style_encoder_from_config(
    cfg: DictConfig,
    device: torch.device,
) -> StyleEncoder | MetaStyleSpeech:
    components, paths = _comps_and_paths_from_config(cfg)

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

    return style_encoder


def dddm_from_config(
    cfg: DictConfig,
    device: torch.device = torch.device("cpu"),
) -> DDDM:
    components, paths = _comps_and_paths_from_config(cfg)

    ## Initialize decoder and diffusion model
    decoder_cls = components["decoder"]
    decoder = decoder_cls(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)
    if path := paths.get(decoder_cls):
        util.load_model(decoder, path)

    diffusion_cls = components["diffusion"]
    diffusion = diffusion_cls(cfg.model.diffusion).to(device)
    if path := paths.get(diffusion_cls):
        util.load_model(diffusion, path)

    ## Finalize and return model components
    return DDDM(decoder, diffusion).to(device)
