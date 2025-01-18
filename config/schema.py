from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING, OmegaConf


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
    def from_yaml(cls, path: str) -> "Config":
        data = OmegaConf.load(path)
        return cls(
            training=TrainingConfig(**data.training),
            dataset=DatasetConfig(**data.dataset),
            dataloader=DataLoaderConfig(**data.dataloader),
        )
