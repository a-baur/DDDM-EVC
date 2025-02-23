import typing
from enum import Enum

import util
from config import DDDMEVCConfig, DDDMVCConfig
from models.content_encoder import XLSR
from models.diffusion import Diffusion
from models.pitch_encoder import VQVAEEncoder
from models.source_filter_encoder import SourceFilterEncoder
from models.style_encoder import MetaStyleSpeech, StyleEncoder
from modules.wavenet_decoder import WavenetDecoder

from .dddm import DDDM


class Pretrained(Enum):
    METASTYLE_SPEECH = "metastylespeech.pth"
    VQVAE = "vqvae.pth"
    VC_WAVENET = "vc/wavenet_decoder.pth"
    VC_DIFFUSION = "vc/diffusion.pth"
    EVC_STYLEENCODER = "evc/style_encoder.pth"
    EVC_WAVENET = "evc/wavenet_decoder.pth"
    EVC_DIFFUSION = "evc/diffusion.pth"


MODEL_BLUEPRINT = {
    DDDMVCConfig: {
        "style_encoder": MetaStyleSpeech,
        "encoder.content_encoder": XLSR,
        "encoder.pitch_encoder": VQVAEEncoder,
        "encoder.decoder": WavenetDecoder,
        "diffusion": Diffusion,
        "pretrained": {
            "encoder.pitch_encoder": Pretrained.VQVAE,
            "encoder.decoder": Pretrained.VC_WAVENET,
            "style_encoder": Pretrained.METASTYLE_SPEECH,
            "diffusion": Pretrained.VC_DIFFUSION,
        },
        "freeze": ["style_encoder", "encoder.content_encoder"],
    },
    DDDMEVCConfig: {
        "style_encoder": StyleEncoder,
        "encoder.content_encoder": XLSR,
        "encoder.pitch_encoder": VQVAEEncoder,
        "encoder.decoder": WavenetDecoder,
        "diffusion": Diffusion,
        "pretrained": {
            "encoder.pitch_encoder": Pretrained.VQVAE,
            "style_encoder.speaker_encoder": Pretrained.METASTYLE_SPEECH,
        },
        "freeze": [
            "style_encoder.speaker_encoder",
            "style_encoder.emotion_encoder",
            "encoder.content_encoder",
        ],
    },
    # D4MEVCConfig: {
    #     "style_encoder": StyleEncoder,
    #     "encoder.content_encoder": XLSR,
    #     "encoder.pitch_encoder": VQVAEEncoder,
    #     "encoder.decoder": WavenetDecoder,
    #     "diffusion": Diffusion,
    #     "pretrained": {
    #         "encoder.pitch_encoder": Pretrained.VQVAE,
    #         "style_encoder.speaker_encoder": Pretrained.METASTYLE_SPEECH,
    #     },
    #     "freeze": [
    #         "style_encoder.speaker_encoder",
    #         "style_encoder.emotion_encoder",
    #         "encoder.content_encoder",
    #     ],
    # },
}


@typing.no_type_check
def dddm_from_config(
    cfg: DDDMVCConfig | DDDMEVCConfig,
    sample_rate: int = 16000,
    pretrained: bool = False,
) -> DDDM:
    """
    Builds DDDM model from configuration.

    :param cfg: ConfigVC or ConfigEVC
    :param sample_rate: Sample rate of data
    :param pretrained: If true, load pretrained models
    :return: DDDM model
    """
    cfg_type = type(cfg)
    if cfg_type not in MODEL_BLUEPRINT:
        raise ValueError(f"Unknown config type: {cfg_type.__name__}")

    blueprint = MODEL_BLUEPRINT[cfg_type]

    # Initialize components dynamically
    content_encoder = blueprint["encoder.content_encoder"]()
    pitch_encoder = blueprint["encoder.pitch_encoder"](cfg.pitch_encoder)
    decoder = blueprint["encoder.decoder"](
        cfg.decoder,
        content_dim=cfg.content_encoder.out_dim,
        f0_dim=cfg.pitch_encoder.vq.k_bins,
    )
    diffusion = blueprint["diffusion"](cfg.diffusion)
    style_encoder = blueprint["style_encoder"](cfg.style_encoder)

    # Build model
    src_ftr_encoder = SourceFilterEncoder(
        content_encoder, pitch_encoder, decoder, sample_rate=sample_rate
    )
    model = DDDM(style_encoder, src_ftr_encoder, diffusion)

    # Load pretrained models if requested
    _load_pretrained_models(blueprint["pretrained"], model)
    if pretrained:
        _freeze_models(blueprint.get("freeze", []), model)

    return model


def _load_pretrained_models(
    pretrained_dict: dict[str, Pretrained], model: DDDM
) -> None:
    """Loads pretrained weights based on provided dictionary."""
    for attr, path_enum in pretrained_dict.items():
        attr_path = attr.split(".")  # Handle nested attributes
        target = model
        for key in attr_path:
            target = getattr(target, key)
        util.load_model(target, path_enum.value)


def _freeze_models(attributes: list[str], model: DDDM) -> None:
    """Freezes all parameters for specified model attributes."""
    for attr in attributes:
        attr_path = attr.split(".")
        target = model
        for key in attr_path:
            target = getattr(target, key)
        target.requires_grad_(False)
