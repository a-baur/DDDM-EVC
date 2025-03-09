from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig

from data import AudioDataloader, MSPPodcast, MSPPodcastFilenames
from models import models_from_config
from models.dddm.preprocessor import DDDMInput, Label


def test_batch_save_load(tmp_path: Path) -> None:
    batch_size = 10
    batch = DDDMInput(
        audio=torch.rand(batch_size, 100),
        mel=torch.rand(batch_size, 80, 100),
        mask=torch.rand(batch_size, 80, 100),
        emb_pitch=torch.rand(batch_size, 100),
        emb_content=torch.rand(batch_size, 100),
        labels=Label(torch.randint(0, 1000, (batch_size, 5))),
    )
    filenames = [f"sample_{i}.pth" for i in range(batch_size)]

    batch.save(tmp_path, filenames)
    loaded = DDDMInput.load(tmp_path, filenames)

    assert batch.audio.eq(loaded.audio).all()
    assert batch.mel.eq(loaded.mel).all()
    assert batch.mask.eq(loaded.mask).all()
    assert batch.emb_pitch.eq(loaded.emb_pitch).all()
    assert batch.emb_content.eq(loaded.emb_content).all()
    assert batch.labels.label_tensor.eq(loaded.labels.label_tensor).all()


@pytest.mark.parametrize("config_name", ["evc_xlsr", "evc_hu"])
def test_feature_extraciton(
    tmp_path: Path,
    model_config: DictConfig,
    device: torch.device,
) -> None:
    _, preprocessor, _ = models_from_config(model_config, device)
    dataset = MSPPodcast(
        model_config.data,
        split="train",
        return_filename=True,
    )
    dataloader = AudioDataloader(
        dataset,
        model_config.data.dataloader,  # noqa
        batch_size=4,
    )

    batch: tuple[torch.Tensor, torch.Tensor, list[str]] = next(iter(dataloader))
    audio, n_frames, fnames = batch[0].to(device), batch[1].to(device), batch[2]
    x = preprocessor(audio, n_frames)
    x.save(tmp_path, fnames)

    dataset = MSPPodcastFilenames(
        model_config.data,
        split="train",
        extract_path=tmp_path,
    )
    dataloader = AudioDataloader(
        dataset,
        model_config.data.dataloader,  # noqa
        batch_size=4,
    )
    paths, fnames = next(iter(dataloader))
    loaded = DDDMInput.load(paths[0], fnames)

    assert x.audio.eq(loaded.audio).all()
    assert x.mel.eq(loaded.mel).all()
    assert x.mask.eq(loaded.mask).all()
    assert x.emb_pitch.eq(loaded.emb_pitch).all()
    assert x.emb_content.eq(loaded.emb_content).all()
    assert x.labels.label_tensor.eq(loaded.labels.label_tensor).all()
