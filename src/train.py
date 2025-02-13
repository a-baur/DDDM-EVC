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

config.register_configs()


class Trainer:
    def __init__(
        self,
        model: DDDM,
        mel_transform: MelTransform,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataloader: AudioDataloader,
        eval_dataloader: AudioDataloader,
        device: torch.device,
        scaler: GradScaler,
        cfg: config.Config,
        distributed: bool,
        rank: int,
    ):
        self.model = model
        self.mel_transform = mel_transform
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.scaler = scaler
        self.cfg = cfg
        self.distributed = distributed
        self.rank = rank

    def train_epoch(self) -> None:
        """

        :return:
        """
        for batch_idx, (x, x_n_frames) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            x_mel = self.mel_transform(x)
            diff_loss, rec_loss = self.model.compute_loss(x, x_mel, x_n_frames)

            loss = (
                diff_loss * self.cfg.training.diff_loss_coef
                + rec_loss * self.cfg.training.rec_loss_coef
            )

            if self.cfg.training.use_fp16_scaling:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = util.clip_grad_value(
                    self.model.parameters(),
                    self.cfg.training.clip_value,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = util.clip_grad_value(
                    self.model.parameters(),
                    self.cfg.training.clip_value,
                )
                self.optimizer.step()

            print(grad_norm)  # logging
            self.scheduler.step()

    def train(self, n_epochs: int) -> None:
        """

        :param n_epochs:
        :return:
        """
        self.model.train()

        for epoch in range(n_epochs):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)  # noqa

            self.train_epoch()


@hydra.main(
    version_base=None,
    config_path=config.CONFIG_PATH.as_posix(),
    config_name="config_vc",
)  # type: ignore
def run(cfg: DictConfig) -> None:
    cfg: config.Config = OmegaConf.to_object(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
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
            backend='nccl', init_method='env://', world_size=n_gpus, rank=rank
        )
    if on_cuda:
        torch.cuda.set_device(rank)

    train_dataset = MSPPodcast(cfg.data, split="train")
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    train_loader = AudioDataloader(train_dataset, cfg=cfg, sampler=train_sampler)

    eval_dataset = MSPPodcast(cfg.data, split="test1")
    eval_loader = AudioDataloader(eval_dataset, cfg=cfg, shuffle=False)

    mel_transform = MelTransform(cfg.data.mel_transform)

    model = DDDM(cfg.model, sample_rate=cfg.data.dataset.sampling_rate)
    model.load_pretrained(
        models=("speaker_encoder", "pitch_encoder"), mode="eval", device=device
    )

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
    run()
