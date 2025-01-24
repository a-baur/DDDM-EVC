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
class SpeakerEncoderConfig:
    in_dim: int
    hidden_dim: int
    out_dim: int


@dataclass
class ModelsConfig:
    speaker_encoder: SpeakerEncoderConfig


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
