import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import config
import util
from data import AudioDataloader, MelTransform, MSPPodcast
from models import DDDM
from util.training import Trainer

config.register_configs()


@hydra.main(
    version_base=None,
    config_path=config.CONFIG_PATH.as_posix(),
    config_name="config_vc",
)  # type: ignore
def main(cfg: DictConfig) -> None:
    cfg: config.Config = OmegaConf.to_object(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        devices = util.get_cuda_devices()
        proceed = input(
            f"proceed training on the following cuda devices (y/n)? {devices}"
        )
        if proceed.lower() == "n":
            return None

    if n_gpus > 1:
        mp.spawn(train, nprocs=n_gpus, args=(device, n_gpus, cfg))
    else:
        train(0, device, n_gpus, cfg)


def setup_trainer(
    rank: int, device: torch.device, n_gpus: int, cfg: config.Config
) -> Trainer:
    on_cuda = n_gpus > 0
    distributed = n_gpus > 1

    if distributed:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
        )
    if on_cuda:
        torch.cuda.set_device(rank)

    train_dataset = MSPPodcast(
        cfg.data, split="development"
    )  # split="train")  TODO: replace for actual training
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = AudioDataloader(train_dataset, cfg=cfg, sampler=train_sampler)

    eval_dataset = MSPPodcast(
        cfg.data, split="development"
    )  # split="train")  TODO: replace for actual training
    eval_loader = AudioDataloader(eval_dataset, cfg=cfg, batch_size=1, shuffle=False)

    mel_transform = MelTransform(cfg.data.mel_transform)

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(
        models=("style_encoder", "pitch_encoder"), mode="eval", device=device
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        cfg.training.learning_rate,
        betas=cfg.training.betas,
        eps=cfg.training.eps,
    )

    if distributed:
        model = DistributedDataParallel(model, device_ids=[rank])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg.training.lr_decay, last_epoch=-1
    )
    scaler = GradScaler(
        device="cuda" if on_cuda else "cpu", enabled=cfg.training.use_fp16_scaling
    )

    return Trainer(
        model=model,
        mel_transform=mel_transform,
        optimizer=optimizer,
        scheduler=scheduler_g,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        device=device,
        scaler=scaler,
        cfg=cfg,
        distributed=distributed,
        rank=rank,
    )


def train(rank: int, device: torch.device, n_gpus: int, cfg: config.Config) -> None:
    torch.manual_seed(cfg.training.seed)
    trainer = setup_trainer(rank, device, n_gpus, cfg)
    trainer.train(cfg.training.epochs)


if __name__ == "__main__":
    os.chdir(util.get_root_path())
    main()
