from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

# Path to the config directory
CONFIG_PATH = Path(__file__).parent.parent / "config"


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    segment_size: int


@dataclass
class DatasetConfig:
    name: str
    path: str
    sampling_rate: int
    segment_size: int
    segment_seed: int


@dataclass
class DataLoaderConfig:
    num_workers: int
    distributed: bool
    pin_memory: bool
    drop_last: Optional[bool]


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
class SourceFilterEncoderConfig:
    speaker_encoder: MetaStyleSpeechConfig
    pitch_encoder: VQVAEConfig
    decoder: WavenetDecoderConfig


@dataclass
class DiffusionConfig:
    dec_dim: int
    spk_dim: int
    use_ref_t: bool
    beta_min: float
    beta_max: float


@dataclass
class ModelsConfig:
    src_ftr_encoder: SourceFilterEncoderConfig
    src_ftr_decoder: DiffusionConfig
    vocoder: HifiGANConfig


@dataclass
class Config:
    training: TrainingConfig
    data: DataConfig
    models: ModelsConfig

    @classmethod
    def from_yaml(cls, name: str) -> "Config":
        """
        Load a Config object from a YAML file
        with type validation.

        :param name: Name of the YAML file
        :return: Config object
        """
        cfg = OmegaConf.structured(Config)
        cfg.merge_with(OmegaConf.load(CONFIG_PATH / name))
        return cls(
            training=TrainingConfig(**cfg.training),
            data=DataConfig(**cfg.data),
            models=ModelsConfig(**cfg.models),
        )
