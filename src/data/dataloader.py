from typing import Callable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torchaudio.transforms import MelSpectrogram

from config import Config, MelTransformConfig


class AudioDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
        sampler: Sampler = None,
        shuffle: bool = False,
        collate_fn: Callable | None = None,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.dataloader.num_workers,
            sampler=sampler,
            pin_memory=cfg.data.dataloader.pin_memory,
            drop_last=cfg.data.dataloader.drop_last,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )


class MelTransform(torch.nn.Module):
    """
    Apply mel transformation to waveform input and add log10 scale.
    """

    def __init__(self, cfg: MelTransformConfig) -> None:
        super().__init__()
        self.mel_transform = MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.filter_length,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mel_channels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            window_fn=torch.hann_window,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.log(self.mel_transform(x) + 0.001)
        return outputs[..., :-1]
