import pathlib

import torch
import torchaudio

import config
from util import get_root_path, random_segment


def librispeech_collate_fn(
    batch: list[tuple[torch.Tensor, ...]],
) -> tuple[torch.Tensor, torch.Tensor]:
    config.register_configs()
    cfg = config.load_hydra_config_vc("config_vc.yaml")

    audio = next(zip(*batch))
    segments = [random_segment(a, cfg.data.dataset.segment_size) for a in audio]
    audio, length = zip(*segments)
    audio = torch.stack(audio)
    audio = audio.squeeze(1)  # (B, 1, T) -> (B, T), mono audio
    # number of frames without padding
    length = torch.tensor(length)
    length = length // cfg.data.mel_transform.hop_length
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
