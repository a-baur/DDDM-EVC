import pathlib

import torch
import torchaudio

from config import Config
from util import get_root_path, random_segment

DATA_CONFIG = Config.from_yaml("config.yaml").data


def librispeech_collate_fn(
    batch: list[tuple[torch.Tensor, ...]]
) -> tuple[torch.Tensor, torch.Tensor]:
    audio = next(zip(*batch))
    segments = [random_segment(a, DATA_CONFIG.dataset.segment_size) for a in audio]
    audio, length = zip(*segments)
    audio = torch.stack(audio)
    audio = audio.squeeze(1)  # (B, 1, T) -> (B, T), mono audio
    length = torch.tensor(length)
    # number of frames without padding
    length = length // DATA_CONFIG.mel_transform.hop_length
    return audio, length


def Librispeech(
    url: str,
    folder_in_archive: str,
    path: str | pathlib.Path = get_root_path(),
) -> torchaudio.datasets.LIBRISPEECH:
    """
    Load the LibriSpeech dataset.

    :param url: URL of the dataset
    :param folder_in_archive: Folder in the archive
    :param path: Path to the root directory (default: project root)
    :return: LibriSpeech dataset
    """
    path = pathlib.Path(path)
    root = path / "datasets" / "librispeech"
    if not root.exists():
        root.mkdir(parents=True)
    ds = torchaudio.datasets.LIBRISPEECH(
        root.as_posix(), url, folder_in_archive, download=True
    )
    return ds
