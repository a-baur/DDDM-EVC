import torch
from omegaconf import DictConfig

import util
from data import MelTransform
from models.content_encoder import XLSR, XLSR_ESPEAK_CTC, Hubert
from models.diffusion import Diffusion
from models.pitch_encoder import VQF0Encoder, YINEncoder
from models.style_encoder import (
    MetaStyleSpeech,
    StyleEncoder,
    StyleLabelEncoder,
    DisentangledStyleEncoder,
)
from modules.wavenet_decoder import WavenetDecoder, WavenetAutostylizedDecoder

from ..speech_token_encoder import SpeechTokenAutoStylizer, SpeechTokenConcatenator
from ..token_diffusion import TokenDiffusion
from .base import DDDM, TokenDDDM
from .preprocessor import BasePreprocessor, DDDMPreprocessor, DurDDDMPreprocessor


def models_from_config(
    config: DictConfig, device: torch.device = None
) -> tuple[
    DDDM | TokenDDDM,
    BasePreprocessor,
    MetaStyleSpeech | StyleEncoder | StyleLabelEncoder,
]:
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
    elif model_choice == "evc_xlsr_disentangled":
        return build_evc_xlsr_disentangled(config, device)
    elif model_choice == "evc_xlsr_yin_disentangled":
        return build_evc_xlsr_yin_disentangled(config, device)
    elif model_choice == "evc_xlsr_ph":
        return build_evc_xlsr_ph(config, device)
    elif model_choice == "evc_xlsr_ph_yin":
        return build_evc_xlsr_ph_yin(config, device)
    elif model_choice == "evc_xlsr_yin":
        return build_evc_xlsr_yin(config, device)
    elif model_choice == "evc_xlsr_yin_label":
        return build_evc_xlsr_yin_label(config, device)
    elif model_choice == "evc_xlsr_yin_l2_norm":
        return build_evc_xlsr_yin(config, device)
    elif model_choice == "tkn_evc_xlsr_yin":
        return build_tkn_evc_xlsr_yin(config, device)
    elif model_choice == "tkn_autostylizer":
        return build_tkn_autostylizer(config, device)
    elif model_choice == "evc_autostylizer":
        return build_evc_autostylizer(config, device)
    else:
        raise ValueError(f"Unknown model configuration: {model_choice}")


def build_vc_xlsr(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, MetaStyleSpeech]:
    """Build DDDM VC XLSR model."""
    style_encoder = MetaStyleSpeech(cfg.model.style_encoder).to(device)

    vq_vae = VQF0Encoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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

    vq_vae = VQF0Encoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR_ESPEAK_CTC().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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

    vq_vae = VQF0Encoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth")

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion, score_model_ver=2).to(device),
    )

    return model, preprocessor, style_encoder


def build_evc_xlsr_disentangled(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, DDDMPreprocessor, StyleEncoder]:
    """Build DDDM EVC XLSR model."""
    style_encoder = DisentangledStyleEncoder(
        cfg.model.style_encoder,
        hidden_dim=256,
        n_spk=1459,
    ).to(device)
    util.load_model(
        style_encoder,
        "disentangled_style_encoder.pth",
        model_key="model",
        mode="eval",
        freeze=True,
    )

    vq_vae = VQF0Encoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth")

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=XLSR().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
        perturb_target=cfg.model.perturb_target,
    )

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion, score_model_ver=1).to(device),
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

    vq_vae = VQF0Encoder(cfg.model.pitch_encoder).to(device)
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
        perturb_target=cfg.model.perturb_target,
        flatten_pitch=cfg.model.flatten_pitch,
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
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=True,
            return_hidden=False,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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

    vq_vae = VQF0Encoder(cfg.model.pitch_encoder).to(device)
    util.load_model(vq_vae, "vqvae.pth", mode="eval", freeze=True)

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=vq_vae,
        content_encoder=Hubert().to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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


def build_evc_xlsr_yin_disentangled(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, StyleEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = DisentangledStyleEncoder(
        cfg.model.style_encoder,
        hidden_dim=256,
        n_spk=1459,
    ).to(device)
    util.load_model(
        style_encoder,
        "disentangled_style_encoder.pth",
        model_key="model",
        mode="eval",
        freeze=True,
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
    ).to(device)

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion, score_model_ver=2),
    ).to(device)

    return model, preprocessor, style_encoder


def build_evc_xlsr_yin(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, StyleEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder,
        "metastylespeech.pth",
        mode="eval",
        freeze=True,
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
    ).to(device)

    src_ftr_encoder = WavenetDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion),
    ).to(device)

    return model, preprocessor, style_encoder


def build_evc_xlsr_yin_label(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, StyleLabelEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = StyleLabelEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder,
        "metastylespeech.pth",
        mode="eval",
        freeze=True,
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=True,
            return_hidden=True,
            logits_to_phoneme=True,
        ).to(device),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
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


def build_tkn_evc_xlsr_yin(
    cfg: DictConfig, device: torch.device
) -> tuple[TokenDDDM, BasePreprocessor, StyleEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder,
        "metastylespeech.pth",
        mode="eval",
        freeze=True,
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform),
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
    ).to(device)

    model = TokenDDDM(
        token_encoder=SpeechTokenConcatenator(cfg.model.token_encoder),
        diffusion=TokenDiffusion(cfg.model.diffusion),
    ).to(device)

    return model, preprocessor, style_encoder


def build_tkn_autostylizer(
    cfg: DictConfig, device: torch.device
) -> tuple[TokenDDDM, BasePreprocessor, StyleEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder,
        "metastylespeech.pth",
        mode="eval",
        freeze=True,
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform),
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
    ).to(device)

    model = TokenDDDM(
        token_encoder=SpeechTokenAutoStylizer(cfg.model.token_encoder),
        diffusion=TokenDiffusion(cfg.model.diffusion),
    ).to(device)

    return model, preprocessor, style_encoder


def build_evc_autostylizer(
    cfg: DictConfig, device: torch.device
) -> tuple[DDDM, BasePreprocessor, StyleEncoder]:
    """Build DDDM VC XLSR with pitch encoder model."""
    style_encoder = StyleEncoder(cfg.model.style_encoder).to(device)
    util.load_model(
        style_encoder.speaker_encoder,
        "metastylespeech.pth",
        mode="eval",
        freeze=True,
    )

    preprocessor = DDDMPreprocessor(
        mel_transform=MelTransform(cfg.data.mel_transform).to(device),
        pitch_encoder=YINEncoder(cfg.model.pitch_encoder).to(device),
        content_encoder=XLSR_ESPEAK_CTC(
            return_logits=False,
            return_hidden=True,
        ),
        sample_rate=cfg.data.dataset.sampling_rate,
        perturb_inputs=cfg.model.perturb_inputs,
        flatten_pitch=cfg.model.flatten_pitch,
    ).to(device)

    src_ftr_encoder = WavenetAutostylizedDecoder(
        cfg.model.decoder,
        content_dim=cfg.model.content_encoder.out_dim,
        pitch_dim=cfg.model.pitch_encoder.out_dim,
    ).to(device)

    model = DDDM(
        encoder=src_ftr_encoder,
        diffusion=Diffusion(cfg.model.diffusion, score_model_ver=2),
    ).to(device)

    return model, preprocessor, style_encoder
