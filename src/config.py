from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

# Path to the config directory
CONFIG_PATH = Path(__file__).parent.parent / "config"


# ---------------------
# Training Configuration
# ---------------------
@dataclass
class TrainingConfig:
    seed: int
    batch_size: int
    epochs: int
    segment_size: int
    learning_rate: float
    lr_decay: float
    betas: tuple[float, float]
    eps: float
    use_fp16_scaling: bool
    src_ftr_loss_coef: float
    score_loss_coef: float
    rec_loss_coef: float
    dur_loss_coef: float
    log_interval: int
    eval_interval: int
    save_interval: int
    output_dir: str
    eval_n_batches: int
    eval_batch_size: int
    checkpoint: Optional[str] = None
    clip_value: Optional[float] = None
    compute_emotion_loss: bool = False


# ---------------------
# Dataset and DataLoader Configuration
# ---------------------
@dataclass
class DatasetConfig:
    name: str
    path: str
    sampling_rate: int
    segment_size: int


@dataclass
class DataLoaderConfig:
    num_workers: int
    distributed: bool
    pin_memory: bool
    drop_last: Optional[bool] = None


@dataclass
class MelTransformConfig:
    sample_rate: int
    filter_length: int
    win_length: int
    hop_length: int
    n_mel_channels: int
    f_min: int
    f_max: int


@dataclass
class DataConfig:
    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    mel_transform: MelTransformConfig


# ---------------------
# Model Configurations
# ---------------------
@dataclass
class MetaStyleSpeechConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int


@dataclass
class Resnet1DConfig:
    depth: int
    m_conv: float
    dilation_growth_rate: int = 1
    dilation_cycle: Optional[int] = None
    reverse_dialation: bool = False
    zero_out: bool = False
    res_scale: bool = False
    checkpoint_res: bool = False


@dataclass
class XLSRConfig:
    out_dim: int


@dataclass
class HubertConfig:
    out_dim: int


@dataclass
class W2VLRobustConfig:
    out_dim: int


@dataclass
class F0VAEConfig:
    in_dim: int
    out_dim: int
    hidden_dim: int
    levels: int
    downs_t: list[int]
    strides_t: list[int]
    resnet1d: Resnet1DConfig


@dataclass
class F0VQConfig:
    k_bins: int
    emb_dim: int
    levels: int
    mu: float


@dataclass
class VQVAEConfig:
    sample_rate: int
    f0_encoder: F0VAEConfig
    vq: F0VQConfig
    out_dim: int


@dataclass
class TokenEncoderConfig:
    pitch_dim: int
    content_dim: int
    gin_channels: int
    out_dim: int


@dataclass
class SpeechTokenAutoencoderConfig:
    pitch_dim: int
    content_dim: int
    gin_channels: int
    out_dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    use_positional_encoding: bool


@dataclass
class YinEncoderConfig:
    sample_rate: int
    win_length: int
    hop_length: int
    fmin: float
    fmax: float
    scope_fmin: float
    scope_fmax: float
    bins: int
    out_dim: int


@dataclass
class WavenetDecoderConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    gin_channels: int
    frame_wise_pitch: bool = False


@dataclass
class HifiGANConfig:
    in_dim: int
    resblock: int
    resblock_kernel_sizes: list[int]
    resblock_dilation_sizes: list[list[int]]
    upsample_rates: list[int]
    upsample_initial_channel: int
    upsample_kernel_sizes: list[int]


@dataclass
class DiffusionConfig:
    in_dim: int
    dec_dim: int
    cond_dim: int
    use_ref_t: bool
    beta_min: float
    beta_max: float
    gin_channels: int


@dataclass
class StyleEncoderConfig:
    emotion_emb_dim: int
    speaker_encoder: MetaStyleSpeechConfig
    emotion_encoder: W2VLRobustConfig
    l2_normalize: bool = False
    p_emo_masking: float = 0.0
    p_spk_masking: float = 0.0


@dataclass
class DDDM_VC_XLSR_Config:
    style_encoder: MetaStyleSpeechConfig
    content_encoder: XLSRConfig
    pitch_encoder: VQVAEConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


@dataclass
class DDDM_VC_XLSR_YIN_Config:
    style_encoder: MetaStyleSpeechConfig
    content_encoder: XLSRConfig
    pitch_encoder: YinEncoderConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


@dataclass
class DDDM_EVC_XLSR_YIN_Config:
    style_encoder: StyleEncoderConfig
    content_encoder: XLSRConfig
    pitch_encoder: YinEncoderConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


@dataclass
class TKN_DDDM_EVC_XLSR_YIN_Config:
    style_encoder: StyleEncoderConfig
    content_encoder: XLSRConfig
    pitch_encoder: YinEncoderConfig
    token_encoder: TokenEncoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


@dataclass
class TKN_DDDM_AUTOSTYLIZER_Config:
    style_encoder: StyleEncoderConfig
    content_encoder: XLSRConfig
    pitch_encoder: YinEncoderConfig
    token_encoder: SpeechTokenAutoencoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


@dataclass
class DDDM_EVC_XLSR_Config:
    style_encoder: StyleEncoderConfig
    content_encoder: XLSRConfig
    pitch_encoder: VQVAEConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


@dataclass
class DDDM_EVC_HUBERT_Config:
    style_encoder: StyleEncoderConfig
    content_encoder: HubertConfig
    pitch_encoder: VQVAEConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig
    perturb_inputs: bool = False
    perturb_target: str = None
    flatten_pitch: bool = False
    use_duration_control: bool = False


# ---------------------
# Hydra Configuration
# ---------------------
@dataclass
class VC_XLSR:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: DDDM_VC_XLSR_Config


@dataclass
class VC_XLSR_YIN:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: DDDM_VC_XLSR_YIN_Config


@dataclass
class EVC_XLSR_YIN:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: DDDM_EVC_XLSR_YIN_Config


@dataclass
class TKN_EVC_XLSR_YIN:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: TKN_DDDM_EVC_XLSR_YIN_Config


@dataclass
class TKN_AUTOSTYLIZER:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: TKN_DDDM_AUTOSTYLIZER_Config


@dataclass
class EVC_XLSR:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: DDDM_EVC_XLSR_Config


@dataclass
class EVC_HUBERT:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: DDDM_EVC_HUBERT_Config


@dataclass
class VC_HUBERT:
    component_id: str
    training: TrainingConfig
    data: DataConfig
    model: DDDM_EVC_HUBERT_Config


# ---------------------
# Generalized Hydra Configuration Loader
# ---------------------
cs = ConfigStore.instance()
cs.store(group="model", name="base_vc_xlsr", node=DDDM_VC_XLSR_Config)
cs.store(group="model", name="base_evc_xlsr", node=DDDM_EVC_XLSR_Config)
cs.store(group="model", name="base_evc_hu", node=DDDM_EVC_HUBERT_Config)
cs.store(group="model", name="base_vc_xlsr_yin", node=DDDM_VC_XLSR_YIN_Config)
cs.store(group="model", name="base_evc_xlsr_yin", node=DDDM_EVC_XLSR_YIN_Config)
cs.store(group="model", name="base_tkn_evc_xlsr_yin", node=TKN_DDDM_EVC_XLSR_YIN_Config)
cs.store(group="model", name="base_tkn_autostylizer", node=TKN_DDDM_AUTOSTYLIZER_Config)
cs.store(group="training", name="base_training", node=TrainingConfig)
cs.store(group="data", name="base_data", node=DataConfig)


def load_hydra_config(model_name: str, overrides: list[str] = None) -> DictConfig:
    """Load and instantiate Hydra configuration."""
    _overrides = [
        "training.output_dir='./outputs'",
        f"model={model_name}",
        f"model_choice={model_name}",
    ]
    _overrides.extend(overrides or [])
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH.as_posix()):
        return compose(config_name="config", overrides=_overrides)
