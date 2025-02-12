from pathlib import Path
from typing import Literal

import torch
import torchaudio

from config import DataConfig
from util import random_segment


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
        cfg: DataConfig,
        split: T_SPLITS,
    ) -> None:
        self.path = Path(cfg.dataset.path)
        self.manifest_dir = self.path / self.MANIFEST_FOLDER
        self.name = cfg.dataset.name
        self.sampling_rate = cfg.dataset.sampling_rate
        self.segment_size = cfg.dataset.segment_size
        self.hop_length = cfg.mel_transform.hop_length

        self.split = split
        self.fnames, self.lengths = self._load_manifest(split)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample from the dataset.

        :param index: Index of the sample
        :return: Tuple of audio and number of frames
        """
        fname = self.fnames[index]
        audio, _ = torchaudio.load(self.path / "Audio" / fname)
        audio = audio.squeeze(0)  # (1, T) -> (T,), mono audio

        mel = torch.Tensor([])  # Placeholder for mel spectrogram

        audio, segment_size = random_segment(audio, segment_size=self.segment_size)
        n_frames = segment_size // self.hop_length  # number of frames without padding

        return audio, mel, n_frames

    def __len__(self) -> int:
        return len(self.fnames)

    def _load_manifest(self, split: T_SPLITS) -> tuple[list[str], list[float]]:
        """
        Load manifest files for the MSP Podcast dataset.

        :param split: Split name
        :return: Tuple of samples and lengths
        """
        fname = self.MANIFEST_FILES[split]
        fpath = (self.manifest_dir / fname).absolute()
        with open(fpath, "r") as f:
            lines = f.readlines()[1:]  # Skip header
        unpacked_tuples = [line.strip().split("\t") for line in lines]
        fnames = [t[0] for t in unpacked_tuples]
        lengths = [float(t[1]) for t in unpacked_tuples]
        return fnames, lengths
