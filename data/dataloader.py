import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config.schema import DataLoaderConfig


def collate_fn(batch: list) -> tuple[torch.Tensor, list[int]]:
    """
    Collate function to pad waveforms to the same length.
    """
    # Pad waveforms to the same length
    max_length = max(sample[0].size(1) for sample in batch)
    padded_waveforms = [
        torch.nn.functional.pad(sample[0], (0, max_length - sample[0].size(1)))
        for sample in batch
    ]

    # Stack waveforms and sample rates
    waveforms = torch.stack(padded_waveforms)
    sample_rates = [sample[1] for sample in batch]
    return waveforms, sample_rates


class AudioDataloader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        cfg: DataLoaderConfig,
    ) -> None:
        sampler = DistributedSampler(dataset) if cfg.distributed else None
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            sampler=sampler,
            drop_last=cfg.drop_last,
            collate_fn=collate_fn,
            shuffle=(sampler is None),
        )
