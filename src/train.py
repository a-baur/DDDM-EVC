import os

import hydra
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

import config
import util
from data import AudioDataloader, MSPPodcast
from models import models_from_config
from util.training import Trainer


@hydra.main(
    version_base=None,
    config_path=config.CONFIG_PATH.as_posix(),
    config_name="dddm_vc_xlsr_ph",
)  # type: ignore
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        devices = util.get_cuda_devices()
        devices = "\n".join(devices)
        proceed = input(
            f"proceed training on the following cuda devices (y/n)?\n{devices}\n"
        )
        if proceed.lower() == "n":
            return None

    if n_gpus > 1:
        mp.spawn(train, nprocs=n_gpus, args=(device, n_gpus, cfg))
    else:
        train(0, device, n_gpus, cfg)


def setup_trainer(
    rank: int,
    device: torch.device,
    n_gpus: int,
    cfg: DictConfig,
) -> Trainer:
    on_cuda = n_gpus > 0
    distributed = n_gpus > 1

    if distributed:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["WORLD_SIZE"] = str(n_gpus)
        os.environ["RANK"] = str(rank)

        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
        )
    if on_cuda:
        torch.cuda.set_device(rank)

    train_dataset = MSPPodcast(cfg.data, split="train")
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = AudioDataloader(
        train_dataset,
        cfg=cfg.data.dataloader,
        batch_size=cfg.training.batch_size,
        sampler=train_sampler,
    )

    eval_dataset = MSPPodcast(cfg.data, split="test1")
    eval_loader = AudioDataloader(
        eval_dataset, cfg=cfg.data.dataloader, batch_size=1, shuffle=False
    )

    model, preprocessor, style_encoder = models_from_config(cfg, device=device)
    model = torch.compile(model, backend="inductor")

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
        preprocessor=preprocessor,
        style_encoder=style_encoder,
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


def train(
    rank: int,
    device: torch.device,
    n_gpus: int,
    cfg: DictConfig,
) -> None:
    torch.manual_seed(cfg.training.seed)
    trainer = setup_trainer(rank, device, n_gpus, cfg)
    trainer.train(cfg.training.epochs)


if __name__ == "__main__":
    os.chdir(util.get_root_path())
    main()
