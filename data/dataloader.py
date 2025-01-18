from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from config.schema import DataLoaderConfig


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    cfg: DataLoaderConfig,
) -> DataLoader:
    sampler = DistributedSampler(dataset) if cfg.distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        drop_last=cfg.drop_last,
        shuffle=(sampler is None),
    )
