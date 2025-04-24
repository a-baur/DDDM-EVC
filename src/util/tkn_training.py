import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import GradScaler

import util
from data import AudioDataloader
from models import HifiGAN, W2V2LRobust
from models.dddm.base import TokenDDDM
from models.dddm.duration_control import DurationControl
from models.dddm.preprocessor import BasePreprocessor, DDDMInput

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
    score_loss: float
    rec_loss: float
    dur_loss: float
    grad_norm: float
    learning_rate: float


@dataclass
class EvalMetrics:
    mel_loss: float
    emo_loss: float
    audio: dict[str, torch.Tensor]
    images: dict[str, np.ndarray | None]


class TknTrainer:
    def __init__(
        self,
        model: TokenDDDM,
        preprocessor: BasePreprocessor,
        style_encoder: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_dataloader: AudioDataloader,
        eval_dataloader: AudioDataloader,
        device: torch.device,
        scaler: GradScaler,
        cfg: DictConfig,
        distributed: bool,
        rank: int,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.style_encoder = style_encoder
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
        self.vocoder.eval()

        if cfg.model.use_duration_control:
            self.duration_control = DurationControl(
                cfg.model.content_encoder.out_dim,
                cfg.model.style_encoder.out_dim,
            ).to(device)
        else:
            self.duration_control = None

        if cfg.training.compute_emotion_loss:
            self.compute_emotion_loss = True
            self.emotion_model = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME)
            self.emotion_model.requires_grad_(False)
            self.emotion_model.to(device)
        else:
            self.compute_emotion_loss = False
            self.emotion_model = None

        self.preprocessor.to(device)
        self.style_encoder.to(device)
        self.model.to(device)
        self.vocoder.to(device)

        self.out_dir = Path(cfg.training.output_dir)
        self.train_writer = SummaryWriter(
            (self.out_dir / "tensorboard" / "train").as_posix()
        )
        self.eval_writer = SummaryWriter(
            (self.out_dir / "tensorboard" / "eval").as_posix()
        )
        self.logger = _setup_logger(rank)

        self.logger.info(
            f"Initialized token trainer [{device.type}:{self.rank} | distributed: {self.distributed}]"
        )
        self.logger.info(f"Logging to '{self.out_dir.as_posix()}'")

    def train(self, n_epochs: int) -> None:
        """
        Start training process

        :param n_epochs: Number of epochs to train.
        :return: None
        """
        self.logger.info(f"Starting training [{n_epochs} epochs]...")

        total_steps = n_epochs * self.n_batches

        ckpt_path: str | Path | None = self.cfg.training.checkpoint
        if ckpt_path == "latest":
            ckpt_path = self._find_latest_checkpoint()
        if ckpt_path:
            start_epoch, start_batch = self.load_checkpoint(ckpt_path)
        else:
            self.logger.info("No checkpoint loaded...")
            start_epoch, start_batch = 0, 0

        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        for epoch in range(start_epoch, n_epochs):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)  # noqa

            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch_idx < start_batch:
                    continue  # skip to relevant batch

                global_step = epoch * self.n_batches + batch_idx

                train_metrics = self._train_batch(batch)

                # log, eval and save only if main on thread
                if self.rank == 0 and global_step != 0:
                    if global_step % self.cfg.training.log_interval == 0:
                        self._log_train(
                            global_step,
                            epoch,
                            batch_idx,
                            train_metrics,
                        )

                    if global_step % self.cfg.training.eval_interval == 0:
                        global_progress = global_step / total_steps
                        torch.cuda.empty_cache()
                        eval_metrics = self.eval()
                        self._log_eval(global_step, global_progress, eval_metrics)

                    if global_step % self.cfg.training.save_interval == 0:
                        self.save_checkpoint(epoch, batch_idx)

            # start next epoch with all batches
            start_batch = 0

    def _train_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> TrainMetrics:
        """
        Train single batch of training data.

        :param batch: Batch, containing waveform and unpadded length of frames
        :return: training metrics
        """
        self.preprocessor.train()
        self.model.train()
        self.style_encoder.train()
        if self.duration_control is not None:
            self.duration_control.train()

        self.optimizer.zero_grad()

        audio, n_frames, labels = (
            batch[0].to(self.device, non_blocking=True),
            batch[1].to(self.device, non_blocking=True),
            batch[2].to(self.device, non_blocking=True),
        )
        x = self.preprocessor(audio, n_frames, labels)
        g = self.style_encoder(x).unsqueeze(-1)

        if self.duration_control is not None:
            x, dur_loss = self.duration_control(x, g, return_loss=True)
        else:
            dur_loss = torch.tensor(0.0)

        if self.distributed:
            score_loss, rec_loss = self.model.module.compute_loss(x, g)
        else:
            score_loss, rec_loss = self.model.compute_loss(x, g)

        loss = (
            score_loss * self.cfg.training.score_loss_coef
            + rec_loss * self.cfg.training.rec_loss_coef
            + dur_loss * self.cfg.training.dur_loss_coef
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
            score_loss=score_loss.item(),
            rec_loss=rec_loss.item(),
            dur_loss=dur_loss.item(),
            grad_norm=grad_norm,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

    @torch.no_grad()  # type: ignore
    def eval(self) -> EvalMetrics:
        """
        Evaluate on the evaluation dataset with batched input.

        :return: evaluation metrics
        """
        self.preprocessor.eval()
        self.model.eval()
        self.style_encoder.eval()
        if self.duration_control is not None:
            self.duration_control.eval()

        max_batches = self.cfg.training.eval_n_batches or len(self.eval_dataloader)

        mel_loss = 0.0
        emo_loss = 0.0
        smpl_img = dict()
        smpl_audio = dict()

        sample_count = 0

        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        for batch_idx, batch in enumerate(self.eval_dataloader):
            audio, n_frames, labels = (
                batch[0].to(self.device, non_blocking=True),  # shape [B, ...]
                batch[1].to(self.device, non_blocking=True),  # shape [B]
                batch[2].to(self.device, non_blocking=True),  # shape [B, ...]
            )

            x = self.preprocessor(audio, n_frames, labels)
            g = self.style_encoder(x).unsqueeze(-1)

            y_mel = self.model(x, g, n_time_steps=50)

            mel_loss += F.l1_loss(x.mel, y_mel, reduction="mean").item()

            if self.compute_emotion_loss:
                y = self.vocoder(y_mel)
                emo_loss += self._compute_emo_loss(x, y.squeeze())
            else:
                y = self.vocoder(y_mel[:5, ...])

            # Generate up to 5 samples total
            for i in range(audio.size(0)):
                if sample_count >= 5:
                    break

                y_wv = y[i]

                smpl_audio[f"gen/audio_{sample_count}"] = y_wv.squeeze()
                smpl_img[f"gen/mel_{sample_count}"] = _plot_spectrogram_to_numpy(
                    [x.mel[i], y_mel[i]], ["x", "dddm"]
                )

                sample_count += 1

            if batch_idx + 1 == max_batches:
                break

        mel_loss /= max_batches

        return EvalMetrics(
            mel_loss=mel_loss,
            emo_loss=emo_loss,
            images=smpl_img,
            audio=smpl_audio,
        )

    def save_checkpoint(self, epoch: int, batch_idx: int) -> None:
        """
        Save the current state of training.

        :param epoch: Current epoch
        :param batch_idx: Index of current batch
        :return: None
        """
        ckpt_path = self.out_dir / "ckpt" / f"ckpt_e{epoch}_b{batch_idx}.pth"
        ckpt_path.parent.mkdir(exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "iteration": batch_idx,
            "model": self.model.state_dict(),
            "preprocessor": self.preprocessor.state_dict(),
            "style_encoder": self.style_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        if self.duration_control is not None:
            ckpt["duration_control"] = self.duration_control.state_dict()

        torch.save(ckpt, ckpt_path)
        self.logger.info(f">>> CHECKPOINT SAVED in {ckpt_path.as_posix()}")

    def load_checkpoint(self, ckpt_path: str | Path) -> tuple[int, int]:
        """
        Load checkpoint.

        :param ckpt_path: Path to checkpoint to be loaded.
        :return: tuple of epoch and batch index
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.preprocessor.load_state_dict(ckpt["preprocessor"])
        self.style_encoder.load_state_dict(ckpt["style_encoder"])
        if self.duration_control is not None:
            self.duration_control.load_state_dict(ckpt["duration_control"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        epoch, batch_idx = ckpt["epoch"], ckpt["iteration"]
        self.logger.info(f">>> CHECKPOINT LOADED from {ckpt_path}")
        return epoch, batch_idx

    def _log_train(
        self,
        global_step: int,
        epoch: int,
        batch_idx: int,
        metrics: TrainMetrics,
    ) -> None:
        """
        Log training metrics.

        :param global_step: Current global step.
        :param epoch: Current epoch
        :param batch_idx: Current index of batch
        :param metrics: Training metrics
        :return: None
        """
        batch_progress = batch_idx / self.n_batches
        self.logger.info(
            f"Epoch {epoch}: {batch_progress:7.2%} {batch_idx:4}/{self.n_batches} "
            f"[{metrics.loss=:.5f}, {metrics.score_loss=:.5f}, {metrics.rec_loss=:.5f}]"
        )
        if not VISUALIZATION:
            return

        losses = {
            "total": metrics.loss,
            "score": metrics.score_loss,
            "rec": metrics.rec_loss,
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
        """
        Log evaluation results.

        :param global_step: Current global step
        :param global_progress: Current progress percentage
        :param metrics: Evaluation metrics
        :return: None
        """
        self.logger.info(
            f">>> EVAL [Training progress: {global_progress:7.2%} "
            f"| DDDM L1: {metrics.mel_loss:.5f} , "
            f"Emotion loss: {metrics.emo_loss:.5f}]"
        )
        if not VISUALIZATION:
            return

        losses = {
            "dddm": metrics.mel_loss,
            "emotion-classification-loss": metrics.emo_loss,
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

    def _find_latest_checkpoint(self) -> Optional[Path]:
        """
        Finds the latest checkpoint file in the output directory.

        :return: Path to last checkpoint. If no checkpoint is found, return None.
        """
        output_dir = self.out_dir.parent.parent

        if not output_dir.exists():
            return None

        date_folders = sorted(output_dir.glob("*"), key=lambda p: p.name)
        time_folders = map(
            lambda x: sorted(x.glob("*"), key=lambda p: p.name), date_folders
        )
        time_folders = [fn for subdir in time_folders for fn in subdir]

        if len(time_folders) < 2:
            # no previous output
            return None

        # find last output with checkpoint
        latest_time_folder = Path()
        for i in range(2, len(time_folders) + 1):
            latest_time_folder = time_folders[-i] / "ckpt"
            if latest_time_folder.exists():
                break

        checkpoint_files = sorted(
            latest_time_folder.glob("ckpt_e*_b*.pth"), key=_extract_epoch_batch
        )
        if not checkpoint_files:
            return None
        return checkpoint_files[-1]

    def _compute_emo_loss(self, x: DDDMInput, y: torch.Tensor) -> float:
        assert x.label is not None, "Label is None. Cannot compute emotion loss."
        # shape [B, (Act, Dom, Val)]
        _, emo = self.emotion_model(y, embeddings_only=False)
        labels = x.label.label_tensor[:, [0, 2, 1]]
        emo_loss = F.mse_loss(emo, labels)
        return emo_loss.item()


def _plot_spectrogram_to_numpy(
    mels: torch.Tensor | list[torch.Tensor],
    subtitle: list[str],
) -> Optional[np.array]:
    """
    Plot spectrogram into a numpy array.

    :param mels: List of mel spectrograms.
    :param subtitle: List of subtitles for each mel spectrogram.
    :return: Array of plot
    """
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


def _extract_epoch_batch(path: Path) -> tuple[int, int]:
    """Extracts epoch and batch index from checkpoint filename."""
    match = re.search(r"ckpt_e(\d+)_b(\d+)\.pth", path.name)
    if match:
        return int(match.group(1)), int(match.group(2))  # (epoch, batch)
    return 0, 0  # Default if no match


def _setup_logger(rank: int) -> logging.Logger:
    """
    Sets up a logger that only prints logs for the main process (rank 0).
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(
        logging.DEBUG if rank == 0 else logging.ERROR
    )  # Suppress logs on non-rank 0 processes

    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
