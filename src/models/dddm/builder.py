import torch
from omegaconf import DictConfig

import util
from data import MelTransform
from models.content_encoder import XLSR, XLSR_ESPEAK_CTC, Hubert
from models.diffusion import Diffusion
from models.pitch_encoder import VQVAEEncoder, YINEncoder
from models.style_encoder import MetaStyleSpeech, StyleEncoder
from modules.wavenet_decoder import WavenetDecoder

from .base import DDDM
from .preprocessor import BasePreprocessor, DDDMPreprocessor, DurDDDMPreprocessor


def models_from_config(
    config: DictConfig, device: torch.device = None
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech | StyleEncoder]:
    """Build DDDM model from Hydra configuration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_choice = config.model_choice
    if model_choice == "vc_xlsr":
        return build_vc_xlsr(config, device)
    elif model_choice == "vc_xlsr_ph":
        return build_vc_xlsr_ph(config, device)
    elif model_choice == "vc_xlsr_ph_yin":
        return build_vc_xlsr_ph_yin(config, device)
    elif model_choice == "vc_xlsr_yin":
        return build_vc_xlsr_yin(config, device)
    elif model_choice == "vc_xlsr_yin_dc":
        return build_vc_xlsr_yin_dc(config, device)
    elif model_choice == "evc_xlsr":
        return build_evc_xlsr(config, device)
    elif model_choice == "evc_xlsr_ph":
        return build_evc_xlsr_ph(config, device)
    elif model_choice == "evc_xlsr_ph_yin":
        return build_evc_xlsr_ph_yin(config, device)
    elif model_choice == "evc_xlsr_yin":
        return build_evc_xlsr_yin(config, device)
    elif model_choice == "evc_hu":
        return build_evc_hubert(config, device)
    else:
        raise ValueError(f"Unknown model configuration: {model_choice}")


def build_vc_xlsr(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech]:
    """Build DDDM VC XLSR model."""
    style_encoder = MetaStyleSpeech(cfg.model.style_encoder).to(device)

    vq_vae = VQVAEEncoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_vc_xlsr_ph(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = MetaStyleSpeech(cfg.model.style_encoder).to(device)
    util.load_model(style_encoder, "metastylespeech.pth", mode="eval", freeze=True)

    vq_vae = VQVAEEncoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR_ESPEAK_CTC().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_vc_xlsr_ph_yin(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = MetaStyleSpeech(cfg.model.style_encoder).to(device)
    util.load_model(style_encoder, "metastylespeech.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder().to(device),
        content_encoder=XLSR_ESPEAK_CTC().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_evc_xlsr(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, DDDMPreprocessor, StyleEncoder]:
    """Build DDDM EVC XLSR model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder, "metastylespeech.pth", mode="eval", freeze=True
    )

    vq_vae = VQVAEEncoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth")

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_evc_xlsr_ph(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, DDDMPreprocessor, StyleEncoder]:
    """Build DDDM EVC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder, "metastylespeech.pth", mode="eval", freeze=True
    )

    vq_vae = VQVAEEncoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=True,
            return_hidden=False,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_evc_xlsr_ph_yin(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, DDDMPreprocessor, StyleEncoder]:
    """Build DDDM EVC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder, "metastylespeech.pth", mode="eval", freeze=True
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder().to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=True,
            return_hidden=False,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_evc_hubert(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, DDDMPreprocessor, StyleEncoder]:
    """Build DDDM EVC Hubert model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder, "metastylespeech.pth", mode="eval", freeze=True
    )

    vq_vae = VQVAEEncoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=Hubert().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_vc_xlsr_yin(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = MetaStyleSpeech(cfg.model.style_encoder).to(device)
    util.load_model(style_encoder, "metastylespeech.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder().to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_evc_xlsr_yin(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, StyleEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder, "metastylespeech.pth", mode="eval", freeze=True
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder().to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder


def build_vc_xlsr_yin_dc(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = MetaStyleSpeech(cfg.model.style_encoder).to(device)
    util.load_model(style_encoder, "metastylespeech.pth", mode="eval", freeze=True)

    preprocessor = DurDDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder().to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=True,
            return_hidden=True,
            logits_to_phoneme=True,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion).to(device),
    )

    return model, preprocessor, style_encoder
