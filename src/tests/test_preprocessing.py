from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader, MSPPodcast
from models import models_from_config
from models.dddm.preprocessor import DDDMInput


@pytest.mark.parametrize("config_name", ["evc_xlsr"])
def test_label_reading(
    tmp_path: Path,
    model_config: DictConfig,
    device: torch.device,
) -> None:
    _, preprocessor, _ = models_from_config(model_config, device)
    dataset = MSPPodcast(
        model_config.data,
        split="train",
        load_labels=True,
    )
    dataloader = AudioDataloader(
        dataset,
        model_config.data.dataloader,  # noqa
        batch_size=32,
    )

    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = next(iter(dataloader))
    audio, n_frames, labels = (
        batch[0].to(device),
        batch[1].to(device),
        batch[2].to(device),
    )
    x = preprocessor(audio, n_frames, labels)
    x = x.to(device)
    assert isinstance(x, DDDMInput)
    assert x.label.label_tensor.shape == (32, 5)
