import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchaudio.transforms import MelSpectrogram

from config import Config, MelTransformConfig

from .datasets import AudioDataset


class AudioDataloader(DataLoader):
    def __init__(
        self,
        dataset: AudioDataset,
        cfg: Config,
    ) -> None:
        if cfg.data.dataloader.distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

        super().__init__(
            dataset,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.dataloader.num_workers,
            sampler=sampler,
            drop_last=cfg.data.dataloader.drop_last,
            shuffle=(sampler is None),
        )


class MelSpectrogramFixed(torch.nn.Module):
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
