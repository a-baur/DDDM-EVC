from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# Path to the config directory
CONFIG_PATH = Path(__file__).parent.parent / "config"

T = TypeVar("T")


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
    diff_loss_coef: float
    rec_loss_coef: float
    log_interval: int
    eval_interval: int
    save_interval: int
    output_dir: str
    checkpoint: Optional[str] = None
    eval_n_batches: Optional[int] = None
    clip_value: Optional[float] = None


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
    f0_encoder: F0VAEConfig
    vq: F0VQConfig


@dataclass
class WavenetDecoderConfig:
    in_dim: int
    hidden_dim: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    n_mel_channels: int
    gin_channels: int


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
class DDDMVCConfig:
    speaker_encoder: MetaStyleSpeechConfig
    pitch_encoder: VQVAEConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig


@dataclass
class StyleEncoderConfig:
    speaker_encoder: MetaStyleSpeechConfig
    emotion_emb_dim: int


@dataclass
class DDDMEVCConfig:
    style_encoder: StyleEncoderConfig
    pitch_encoder: VQVAEConfig
    decoder: WavenetDecoderConfig
    diffusion: DiffusionConfig
    vocoder: HifiGANConfig


# ---------------------
# Configuration Wrappers
# ---------------------
@dataclass
class ConfigVC:
    training: TrainingConfig
    data: DataConfig
    model: DDDMVCConfig

    @classmethod
    def from_yaml(cls, name: str) -> "ConfigVC":
        """Load a ConfigVC object from a YAML file with type validation."""
        cfg = OmegaConf.structured(ConfigVC)
        cfg.merge_with(OmegaConf.load(CONFIG_PATH / name))
        return cls(
            training=TrainingConfig(**cfg.training),
            data=DataConfig(**cfg.data),
            model=DDDMVCConfig(**cfg.model),
        )


@dataclass
class ConfigEVC:
    training: TrainingConfig
    data: DataConfig
    model: DDDMEVCConfig

    @classmethod
    def from_yaml(cls, name: str) -> "ConfigEVC":
        """Load a ConfigEVC object from a YAML file with type validation."""
        cfg = OmegaConf.structured(ConfigEVC)
        cfg.merge_with(OmegaConf.load(CONFIG_PATH / name))
        return cls(
            training=TrainingConfig(**cfg.training),
            data=DataConfig(**cfg.data),
            model=DDDMEVCConfig(**cfg.model),
        )


# ---------------------
# Hydra Configuration Registration
# ---------------------
def register_configs() -> None:
    """Register all configurations with Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(name="base_vc", node=ConfigVC)
    cs.store(group="training", name="base_vc", node=TrainingConfig)
    cs.store(group="model", name="base_vc", node=DDDMVCConfig)
    cs.store(group="data", name="base_vc", node=DataConfig)

    cs.store(name="base_evc", node=ConfigEVC)
    cs.store(group="training", name="base_evc", node=TrainingConfig)
    cs.store(group="model", name="base_evc", node=DDDMEVCConfig)
    cs.store(group="data", name="base_evc", node=DataConfig)


# ---------------------
# Generalized Hydra Configuration Loader
# ---------------------
def load_hydra_config(config_name: str, _: Type[T]) -> T:
    """Load and instantiate Hydra configuration."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_PATH.as_posix()):
        dict_cfg = compose(
            config_name=config_name, overrides=["training.output_dir='./outputs'"]
        )
        return OmegaConf.to_object(dict_cfg)  # Convert to Python object


# Convenience wrappers for specific configs
def load_hydra_config_vc(config_name: str) -> ConfigVC:
    return load_hydra_config(config_name, ConfigVC)


def load_hydra_config_evc(config_name: str) -> ConfigEVC:
    return load_hydra_config(config_name, ConfigEVC)
