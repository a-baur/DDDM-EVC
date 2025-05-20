import csv
from pathlib import Path
from typing import Literal, no_type_check

import torch
import torchaudio

import config
import util
from util import random_segment


class MSPPodcast(torch.utils.data.Dataset):
    """
    Dataset class for the MSP Podcast dataset.
    https://doi.org/10.1109/TAFFC.2017.2736999

    Available labels:

    - EmoAct: Emotion Act
    - EmoVal: Emotion Valence
    - EmoDom: Emotion Dominance
    - SpkrID: Speaker ID
    - Gender: Speaker Gender

    :param cfg: DatasetConfig object
    :param split: Split name
    :param random_segmentation: Whether to use random segmentation
    :param load_labels: Whether to load labels
    :param label_filter: Dictionary of labels to filter by.
        E.g. {"EmoAct": 7, "EmoVal": 0.5}
    """

    MANIFEST_FOLDER = "Manifests"
    LABELS_FOLDER = "Labels"
    LABEL_ORDER = ["EmoAct", "EmoVal", "EmoDom", "SpkrID", "Gender"]
    MANIFEST_FILES = {
        "development": "manifest_file_development.txt",
        "test1": "manifest_file_test1.txt",
        "test2": "manifest_file_test2.txt",
        "train": "manifest_file_train.txt",
    }
    T_SPLITS = Literal["development", "test1", "test2", "train"]

    def __init__(
        self,
        cfg: config.DataConfig,
        split: T_SPLITS,
        random_segmentation: bool = True,
        load_labels: bool = True,
        label_filter: dict[str, ...] = None,
    ) -> None:
        assert not (label_filter and not load_labels), (
            "Label filter requires loading labels."
        )

        self.path = Path(cfg.dataset.path)
        if not self.path.is_absolute():
            self.path = Path(util.get_root_path()) / self.path
        self.manifest_dir = self.path / self.MANIFEST_FOLDER
        self.name = cfg.dataset.name
        self.sampling_rate = cfg.dataset.sampling_rate
        self.segment_size = cfg.dataset.segment_size
        self.hop_length = cfg.mel_transform.hop_length

        self.split = split
        self.fnames, self.lengths = self._load_manifest(split)
        self.random_segmentation = random_segmentation
        self.load_labels = load_labels

        if self.load_labels:
            self._load_labels()
        if label_filter:
            self.fnames, self.lengths = self._filter_fnames(label_filter)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, torch.Tensor]:
        """
        Get a sample from the dataset.

        :param index: Index of the sample
        :return: Tuple of audio and number of frames
        """
        fname = self.fnames[index]
        audio, _ = torchaudio.load(self.path / "Audio" / fname)
        audio = audio.squeeze(0)  # (1, T) -> (T,), mono audio

        if self.random_segmentation:
            audio, segment_size = random_segment(audio, segment_size=self.segment_size)
        else:
            segment_size = audio.size(-1)
        n_frames = segment_size // self.hop_length  # number of frames without padding

        if self.load_labels:
            label = self._get_label_for_sample(fname)
            return audio, n_frames, label

        return audio, n_frames

    def __len__(self) -> int:
        return len(self.fnames)

    def _filter_fnames(self, label_filter) -> tuple[list[str], list[float]]:
        """
        Get list of samples for labels.

        :return: List of samples
        """

        def filter_label(fname: str) -> bool:
            for key, value in label_filter.items():
                if isinstance(value, list):
                    if self._transform_label(fname)[key] not in value:
                        return False
                else:
                    if int(self._transform_label(fname)[key]) != value:
                        return False
            return True

        res = list(filter(lambda x: filter_label(x[0]), zip(self.fnames, self.lengths)))
        if not res:
            raise ValueError(f"No samples found for labels: {label_filter}")
        fnames, lengths = zip(*res)
        return list(fnames), list(lengths)

    def _transform_label(self, fname: str) -> dict[str, ...]:
        if fname not in self.labels:
            raise ValueError(f"Label for {fname} not found.")
        label = self.labels[fname]
        label["Gender"] = 0 if label["Gender"] == "Male" else 1
        label["SpkrID"] = -1 if label["SpkrID"] == "Unknown" else int(label["SpkrID"])
        return {k: float(label[k]) for k in self.LABEL_ORDER}

    def _get_label_for_sample(self, fname: str) -> torch.Tensor:
        """
        Get label tensor for a given filename.

        Ordering reflects slicing indices in Label class.

        :param fname: Filename
        :return: Label dictionary
        """
        label = self._transform_label(fname)
        return torch.Tensor(list(label.values()))

    def _load_labels(self) -> None:
        fname = self.path / self.LABELS_FOLDER / "labels_consensus.csv"
        with open(fname, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.labels = {row.pop("FileName"): row for row in reader}

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


class MSPPodcastFilenames(MSPPodcast):
    @no_type_check
    def __init__(self, *args, extract_path: str | Path = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if extract_path is None:
            self.extract_path = self.path / "extracted"
        else:
            self.extract_path = extract_path

    @no_type_check
    def __getitem__(self, index: int) -> tuple[str, str]:
        return (
            self.extract_path.as_posix(),
            self.fnames[index].replace(".wav", ".pth"),
        )
