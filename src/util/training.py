import os

import torch
from torch import GradScaler
from torch.utils.tensorboard import SummaryWriter

import config
import util
from data import AudioDataloader, MelTransform
from models import DDDM


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

        log_dir = cfg.training.tensorboard_dir
        self.train_writer = SummaryWriter(os.path.join(log_dir, "train"))
        self.eval_writer = SummaryWriter(os.path.join(log_dir, "eval"))

    def train(self, n_epochs: int) -> None:
        """

        :param n_epochs:
        :return:
        """
        self.model.train()

        for epoch in range(n_epochs):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)  # noqa

            self.train_epoch(epoch)

    def train_epoch(self, epoch: int) -> None:
        """

        :return:
        """
        n_batches = len(self.train_dataloader)
        for batch_idx, batch in enumerate(self.train_dataloader):
            x, x_n_frames = batch
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

            self.scheduler.step()

            global_step = epoch * n_batches + batch_idx
            self._log_train(global_step, loss, diff_loss, rec_loss, grad_norm)

    def _log_train(
        self,
        step: int,
        total_loss: torch.Tensor,
        diff_loss: torch.Tensor,
        rec_loss: torch.Tensor,
        grad_norm: float,
    ) -> None:
        if self.rank != 0:
            return
        losses = {
            "loss/total": total_loss.item(),
            "loss/rec": rec_loss.item(),
            "loss/diff": diff_loss.item(),
        }
        lr = self.optimizer.param_groups[0]["lr"]
        self.train_writer.add_scalars("loss", losses, step)
        self.train_writer.add_scalar("lr", lr, step)
        self.train_writer.add_scalar("grad_norm", grad_norm)
