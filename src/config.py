from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from omegaconf import MISSING, OmegaConf

# Path to the config directory
CONFIG_PATH = Path(__file__).parent.parent / "config"


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10


@dataclass
class DatasetConfig:
    name: str = "LibriSpeech"
    path: str = MISSING


@dataclass
class DataLoaderConfig:
    num_workers: int
    distributed: bool
    pin_memory: bool
    drop_last: Optional[bool] = False


@dataclass
class Config:
    training: TrainingConfig
    dataset: DatasetConfig
    dataloader: DataLoaderConfig

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
            dataset=DatasetConfig(**cfg.dataset),
            dataloader=DataLoaderConfig(**cfg.dataloader),
        )
