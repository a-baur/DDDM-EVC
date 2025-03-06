from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import torch
from torch import nn

import util
from data import MelTransform
from models.content_encoder import XLSR, Hubert
from models.pitch_encoder import VQVAEEncoder


@dataclass
class Label:
    """
    Dataclass for DDDM input metadata.
    """

    label_tensor: torch.Tensor

    def __post_init__(self) -> None:
        self.emo_act = self.label_tensor[:, 0]
        self.emo_val = self.label_tensor[:, 1]
        self.emo_dom = self.label_tensor[:, 2]
        self.spk_id = self.label_tensor[:, 3]
        self.spk_gender = self.label_tensor[:, 4]


@dataclass
class DDDMInput:
    """
    Dataclass for DDDM input.

    :param audio: Audio waveform
    :param mel: Mel-spectrogram
    :param mask: Padding mask for the mel-spectrogram
    :param emb_pitch: Pitch embedding
    :param emb_content: Content embedding
    :param labels: Metadata labels
    """

    audio: torch.Tensor
    mel: torch.Tensor
    mask: torch.Tensor
    emb_pitch: torch.Tensor
    emb_content: torch.Tensor
    labels: Label

    def __getitem__(self, idx: int) -> list[torch.Tensor]:
        """Get a single sample from the batch."""
        return [
            self.audio[idx : idx + 1],
            self.mel[idx : idx + 1],
            self.mask[idx : idx + 1],
            self.emb_pitch[idx : idx + 1],
            self.emb_content[idx : idx + 1],
            self.labels.label_tensor[idx : idx + 1],
        ]

    def __len__(self) -> int:
        """Get the number of samples in the batch."""
        return self.batch_size

    def __iter__(self) -> Generator[list[torch.Tensor], None, None]:
        """Iterate over the samples in the batch."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        """Get a string representation of the object."""
        return (
            f"DDDMInput(audio={self.audio.size()}, mel={self.mel.size()}, "
            f"mask={self.mask.size()}, emb_pitch={self.emb_pitch.size()}, "
            f"emb_content={self.emb_content.size()})"
        )

    @property
    def device(self) -> torch.device:
        """
        Get the device of the input.

        :return: Device of the input
        """
        return self.audio.device

    @property
    def batch_size(self) -> int:
        """
        Get the batch size of the input.

        :return: Batch size
        """
        return self.audio.size(0)

    def to(self, device: torch.device) -> "DDDMInput":
        """Move tensors in place to the specified device."""
        self.audio = self.audio.to(device, non_blocking=True)
        self.mel = self.mel.to(device, non_blocking=True)
        self.mask = self.mask.to(device, non_blocking=True)
        self.emb_pitch = self.emb_pitch.to(device, non_blocking=True)
        self.emb_content = self.emb_content.to(device, non_blocking=True)
        return self

    @torch.no_grad()
    def save(self, path: str | Path, filenames: list[str]) -> None:
        """
        Save each sample in the input batch to a separate file.

        All features are flattened and saved into a single tensor
        for efficient storage. Offsets and shapes are saved to
        reconstruct the original tensors.

        :param path: Directory to save the files
        :param filenames: List of filenames, one for each sample
        """
        assert len(filenames) == self.batch_size

        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        for tensors, fname in zip(self, filenames):
            shapes = [t.shape for t in tensors]
            dtypes = [t.dtype for t in tensors]
            offsets = [0]
            for t in tensors:
                offsets.append(offsets[-1] + t.numel())

            merged_tensor = torch.cat([t.flatten() for t in tensors])
            out_path = (path / fname).with_suffix(".pth")
            torch.save(
                {
                    "tensor": merged_tensor,
                    "offsets": offsets,
                    "shapes": shapes,
                    "dtypes": dtypes,
                },
                out_path,
            )

    @classmethod
    @torch.no_grad()
    def load(
        cls,
        path: str | Path,
        filenames: list[str],
        pin_memory: bool = False,
    ) -> "DDDMInput":
        """Load the input from a file."""
        if isinstance(path, str):
            path = Path(path)

        file_tensors = []
        batch_offsets = []
        batch_shapes = []
        batch_dtypes = []
        for fname in filenames:
            data = torch.load(path / fname)
            merged_tensor = data["tensor"]
            batch_offsets.append(data["offsets"])
            batch_shapes.append(data["shapes"])
            batch_dtypes.append(data["dtypes"])
            file_tensors.append(merged_tensor)

        assert all(o == batch_offsets[0] for o in batch_offsets)
        assert all(s == batch_shapes[0] for s in batch_shapes)
        assert all(d == batch_dtypes[0] for d in batch_dtypes)

        batch_tensors = torch.stack(file_tensors, dim=0)  # (B, N)
        batch_size = len(filenames)

        shapes = [(batch_size,) + s[1:] for s in batch_shapes[0]]
        offsets = batch_offsets[0]
        dtypes = batch_dtypes[0]
        tensors = [
            batch_tensors[:, offsets[i] : offsets[i + 1]]
            .reshape(shapes[i])
            .to(dtype=dtypes[i])
            for i in range(len(shapes))
        ]
        if pin_memory:
            tensors = [t.pin_memory() for t in tensors]

        return cls(
            audio=tensors[0],
            mel=tensors[1],
            mask=tensors[2],
            emb_pitch=tensors[3],
            emb_content=tensors[4],
            labels=Label(tensors[5]),
        )


class DDDMPreprocessor(nn.Module):
    def __init__(
        self,
        mel_transform: MelTransform,
        pitch_encoder: VQVAEEncoder,
        content_encoder: XLSR | Hubert,
        sample_rate: int,
    ) -> None:
        super().__init__()
        self.mel_transform = mel_transform
        self.pitch_encoder = pitch_encoder
        self.content_encoder = content_encoder
        self.sample_rate = sample_rate

    def __call__(
        self,
        audio: torch.Tensor,
        n_frames: torch.Tensor | None = None,
    ) -> DDDMInput:
        return super().__call__(audio, n_frames)

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        n_frames: torch.Tensor | None = None,
        labels: Label | None = None,
    ) -> DDDMInput:
        """
        Preprocess the audio waveform.

        :param audio: Audio waveform
        :param n_frames: Number of unpaded frames in the mel-spectrogram
        :param labels: Metadata labels
        :return: DDDMInput object
        """
        mel = self.mel_transform(audio)

        if n_frames is None:
            n_frames = torch.full(
                (mel.size(0),), mel.size(2), dtype=torch.long, device=mel.device
            )
        mask = util.sequence_mask(n_frames, mel.size(2)).to(mel.dtype)

        emb_pitch = self.pitch_encoder(audio)

        # ensure xlsr/hubert embedding and x_mask are aligned
        x_pad = util.pad_audio_for_xlsr(audio, self.sample_rate)
        emb_content = self.content_encoder(x_pad)

        if not labels:
            labels = Label(torch.full((audio.size(0), 5), -1).detach().to(audio.device))

        return DDDMInput(
            audio=audio,
            mel=mel,
            mask=mask.detach(),
            emb_pitch=emb_pitch.detach(),
            emb_content=emb_content.detach(),
            labels=labels,
        )
