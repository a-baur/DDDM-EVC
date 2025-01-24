from pathlib import Path
from typing import Literal

import torch
import torchaudio

from config import DatasetConfig


class MSPPodcast(torch.utils.data.Dataset):
    """
    Dataset class for the MSP Podcast dataset.
    https://doi.org/10.1109/TAFFC.2017.2736999

    :param cfg: DatasetConfig object
    """

    MANIFEST_FOLDER = "Manifests"
    MANIFEST_FILES = {
        "development": "manifest_file_development.txt",
        "test1": "manifest_file_test1.txt",
        "test2": "manifest_file_test2.txt",
        "train": "manifest_file_train.txt",
    }
    T_SPLITS = Literal["development", "test1", "test2", "train"]

    def __init__(
        self,
        cfg: DatasetConfig,
        split: T_SPLITS,
    ) -> None:
        self.path = Path(cfg.path)
        self.name = cfg.name
        self.sampling_rate = cfg.sampling_rate
        self.segment_size = cfg.segment_size
        self.segment_seed = cfg.segment_seed

        self.split = split
        self.fnames, self.lengths = self._load_manifest(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        """
        Get a sample from the dataset.

        :param index: Index of the sample
        :return: Tuple of audio and length. Audio is a tensor of shape (segment_size,).
        """
        fname = self.fnames[index]
        length = self.lengths[index]
        audio, _ = torchaudio.load(self.path / "Audio" / fname)
        audio = audio.squeeze(0)  # Remove channel dimension (mono audio)

        audio_size = audio.size(0)
        if audio_size > self.segment_size:
            audio = self._get_random_segment(audio, length)
        else:
            audio = self._get_padded_segment(audio, length)

        return audio, length

    def __len__(self) -> int:
        return len(self.fnames)

    def _load_manifest(self, split: T_SPLITS) -> tuple[list[str], list[float]]:
        """
        Load manifest files for the MSP Podcast dataset.

        :param split: Split name
        :return: Tuple of samples and lengths
        """
        manifest_dir = self.path / self.MANIFEST_FOLDER

        fname = self.MANIFEST_FILES[split]
        with open(manifest_dir / fname, "r") as f:
            lines = f.readlines()[1:]  # Skip header
        unpacked_tuples = [line.strip().split("\t") for line in lines]
        fnames = [t[0] for t in unpacked_tuples]
        lengths = [float(t[1]) for t in unpacked_tuples]
        return fnames, lengths

    def _get_padded_segment(self, audio: torch.Tensor, length: float) -> torch.Tensor:
        """
        Pad the audio segment if it is shorter than the segment size.

        :param audio: Audio tensor
        :param length: Length of the audio
        :return: Padded audio tensor
        """
        return torch.nn.functional.pad(
            audio,
            (0, self.segment_size - audio.size(0)),
            mode="constant",
            value=0,
        )

    def _get_random_segment(self, audio: torch.Tensor, length: float) -> torch.Tensor:
        """
        Randomly segment the audio.

        :param audio: Audio tensor
        :param length: Length of the audio
        :return: Segmented audio tensor
        """
        start = torch.randint(
            low=0,
            high=audio.size(0) - self.segment_size,
            size=(1,),
        ).item()
        audio = audio[start : start + self.segment_size]  # noqa
        return audio
