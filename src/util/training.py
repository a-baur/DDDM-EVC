import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import GradScaler

import config
import util
from data import AudioDataloader, MelTransform
from models import DDDM, HifiGAN

try:
    import matplotlib
    import matplotlib.pylab as plt
    from torch.utils.tensorboard import SummaryWriter

    matplotlib.use("Agg")
    VISUALIZATION = True
except ModuleNotFoundError:
    warnings.warn(
        "Visualization packages not available, "
        "install using 'poetry install --with visualize'"
    )
    VISUALIZATION = False


@dataclass
class TrainMetrics:
    loss: float
    diff_loss: float
    rec_loss: float
    grad_norm: float
    learning_rate: float


@dataclass
class EvalMetrics:
    mel_loss: float
    enc_loss: float
    audio: dict[str, torch.Tensor]
    images: dict[str, np.ndarray | None]


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

        self.n_batches = len(self.train_dataloader)
        self.vocoder = HifiGAN(self.cfg.model.vocoder)
        util.load_model(self.vocoder, "hifigan.pth", device, freeze=True)

        self.model.to(device)
        self.vocoder.to(device)
        self.mel_transform.to(device)

        log_dir = cfg.training.tensorboard_dir
        self.train_writer = SummaryWriter(os.path.join(log_dir, "train"))
        self.eval_writer = SummaryWriter(os.path.join(log_dir, "eval"))
        self.logger = util.setup_logging(os.path.join(log_dir, "training.log"))

        self.logger.info(
            f"Initialized trainer [{device.type}:{self.rank} | distributed: {self.distributed}]"
        )
        self.logger.info(f"Logging to '{log_dir}'")

    def train(self, n_epochs: int) -> None:
        """

        :param n_epochs:
        :return:
        """
        self.logger.info(f"Starting training [{n_epochs} epochs]...")

        total_steps = n_epochs * self.n_batches
        batch: tuple[torch.Tensor, torch.Tensor]
        for epoch in range(n_epochs):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)  # noqa

            for batch_idx, batch in enumerate(self.train_dataloader):
                global_step = epoch * self.n_batches + batch_idx

                train_metrics = self.train_batch(batch)

                # log and eval only if main on thread
                if self.rank != 0:
                    continue

                if global_step % self.cfg.training.log_interval == 0:
                    self._log_train(
                        global_step,
                        epoch,
                        batch_idx,
                        train_metrics,
                    )

                if global_step % self.cfg.training.eval_interval == 0:
                    global_progress = global_step / total_steps
                    eval_metrics = self.eval()
                    self._log_eval(global_step, global_progress, eval_metrics)

    def train_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> TrainMetrics:
        """
        Train single batch of training data.

        :param batch: Batch, containing waveform and unpadded length of frames
        :return: training metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        x, x_n_frames = (
            batch[0].to(self.device, non_blocking=True),
            batch[1].to(self.device, non_blocking=True),
        )
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

        return TrainMetrics(
            loss=loss.item(),
            diff_loss=diff_loss.item(),
            rec_loss=rec_loss.item(),
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    @torch.no_grad()  # type: ignore
    def eval(self) -> EvalMetrics:
        self.model.eval()

        max_batches = self.cfg.training.eval_n_batches
        if not max_batches:
            max_batches = len(self.eval_dataloader)

        mel_loss = 0.0
        enc_loss = 0.0
        images = dict()
        audio = dict()

        batch: tuple[torch.Tensor, torch.Tensor]
        for batch_idx, batch in enumerate(self.eval_dataloader):
            x, x_n_frames = (
                batch[0].to(self.device, non_blocking=True),
                batch[1].to(self.device, non_blocking=True),
            )
            x_mel = self.mel_transform(x)

            y_mel, src_mel, ftr_mel = self.model(
                x, x_mel, x_n_frames, n_time_steps=6, return_enc_out=True
            )
            rec_mel = src_mel + ftr_mel

            mel_loss += F.l1_loss(x_mel, y_mel).item()
            enc_loss += F.l1_loss(x_mel, rec_mel).item()

            if batch_idx < 5:
                # keep track of first five samples in eval dataset
                y = self.vocoder(y_mel)
                enc_wv = self.vocoder(rec_mel)
                src_wv = self.vocoder(src_mel)
                ftr_wv = self.vocoder(ftr_mel)

                audio[f"gen/audio_{batch_idx}"] = y.squeeze()
                audio[f"gen/audio_enc_{batch_idx}"] = enc_wv.squeeze()
                audio[f"gen/audio_src_{batch_idx}"] = src_wv.squeeze()
                audio[f"gen/audio_ftr_{batch_idx}"] = ftr_wv.squeeze()

                images[f"gen/mel_{batch_idx}"] = _plot_spectrogram_to_numpy(
                    [x_mel, y_mel, rec_mel, ftr_mel, src_mel],
                    ["x", "dddm", "encoder", "filter", "source"],
                )

            if batch_idx == max_batches:
                break

        mel_loss /= max_batches
        enc_loss /= max_batches

        return EvalMetrics(
            mel_loss=mel_loss,
            enc_loss=enc_loss,
            images=images,
            audio=audio,
        )

    def _log_train(
        self,
        global_step: int,
        epoch: int,
        batch_idx: int,
        metrics: TrainMetrics,
    ) -> None:
        batch_progress = batch_idx / self.n_batches
        self.logger.info(
            f"Epoch {epoch}: {batch_progress:7.2%} {batch_idx:4}/{self.n_batches} "
            f"[{metrics.loss=:.5f}, {metrics.diff_loss=:.5f}, {metrics.rec_loss=:.5f}]"
        )
        if not VISUALIZATION:
            return

        losses = {
            "total": metrics.loss,
            "reconstruction": metrics.rec_loss,
            "diffusion": metrics.diff_loss,
        }

        self.train_writer.add_scalars("loss", losses, global_step)
        self.train_writer.add_scalar(
            "learning rate", metrics.learning_rate, global_step
        )
        self.train_writer.add_scalar("gradient norm", metrics.grad_norm, global_step)

    def _log_eval(
        self,
        global_step: int,
        global_progress: float,
        metrics: EvalMetrics,
    ) -> None:
        self.logger.info(
            f">>> EVAL [Training progress: {global_progress:4.0%} "
            f"| DDDM L1: {metrics.mel_loss:.5f} , Encoder L1: {metrics.enc_loss:.5f}]"
        )
        if not VISUALIZATION:
            return

        losses = {
            "dddm": metrics.mel_loss,
            "source-filter encoder": metrics.enc_loss,
        }

        self.eval_writer.add_scalars("loss", losses, global_step)

        for tag, image in metrics.images.items():
            self.eval_writer.add_images(
                tag, image, global_step=global_step, dataformats="HWC"
            )

        for tag, audio in metrics.audio.items():
            self.eval_writer.add_audio(
                tag,
                audio,
                global_step=global_step,
                sample_rate=self.cfg.data.dataset.sampling_rate,
            )


def _plot_spectrogram_to_numpy(
    mels: torch.Tensor | list[torch.Tensor],
    subtitle: list[str],
) -> Optional[np.array]:
    if not VISUALIZATION:
        return None

    if not isinstance(mels, list):
        mels = [mels]
    n_mels = len(mels)
    fig, axes = plt.subplots(1, n_mels, figsize=(5 * n_mels, 5), layout="compressed")
    if n_mels == 1:
        axes = [axes]
    im = None
    for i, mel in enumerate(mels):
        mel = mel.clip(min=-10, max=10).squeeze().detach().cpu().numpy()
        im = axes[i].imshow(mel, aspect="auto", origin="lower", vmin=-10, vmax=10)
        axes[i].set_title(subtitle[i])
        axes[i].set_xlabel("Frames")
        axes[i].set_ylabel("Channels")
    plt.colorbar(im, ax=axes)

    # Save the figure directly to a buffer
    fig.canvas.draw()
    buf = fig.canvas.renderer.buffer_rgba()  # noqa

    # Convert buffer to NumPy array
    data = np.frombuffer(buf.tobytes(), dtype=np.uint8)
    width, height = fig.get_size_inches() * fig.dpi
    data = data.reshape(int(height), int(width), 4)  # RGB format
    plt.close()

    return data
