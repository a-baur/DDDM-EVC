import pytest
import torch

from config import Config
from data import AudioDataloader, MSPPodcast, load_librispeech

CONFIG_NAME = "config.yaml"
TESTING_DATASET = "librispeech"  # "msp-podcast"


@pytest.fixture()  # type: ignore
def config() -> Config:
    return Config.from_yaml(CONFIG_NAME)


def _collate_fn(batch):
    # apply segmentize to each audio
    audio = next(zip(*batch))
    audio = [_segmentize(a) for a in audio]
    audio = torch.stack(audio)
    return audio, torch.tensor([a.size(0) for a in audio])


def _segmentize(audio: torch.Tensor, segment_size: int = 38000) -> torch.Tensor:
    audio = audio.squeeze(0)  # Remove channel dimension (mono audio)
    audio_size = audio.size(0)
    if audio_size > segment_size:
        start = torch.randint(
            low=0,
            high=audio.size(0) - segment_size,
            size=(1,),
        ).item()
        audio = audio[start : start + segment_size]  # noqa
    else:
        audio = torch.nn.functional.pad(
            audio,
            (0, segment_size - audio.size(0)),
            mode="constant",
            value=0,
        )

    return audio


@pytest.fixture()  # type: ignore
def dataloader(config: Config) -> AudioDataloader:
    if TESTING_DATASET == "librispeech":
        dataset = load_librispeech("dev-clean", "LibriSpeech")
    elif TESTING_DATASET == "msp-podcast":
        dataset = MSPPodcast(config.data.dataset, split="development")
    else:
        raise ValueError(f"Unknown dataset: {TESTING_DATASET}")
    dataloader = AudioDataloader(dataset=dataset, cfg=config, collate_fn=_collate_fn)
    return dataloader
