from dataclasses import dataclass

import torch
from torch import nn

import util
from data import MelTransform
from models.content_encoder import XLSR, XLSR_ESPEAK_CTC, Hubert
from models.pitch_encoder import VQVAEEncoder, YINEncoder
from util.audio import PraatProcessor


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
    :param phonemes: Phoneme sequence prediction
    :param labels: Metadata labels
    """

    audio: torch.Tensor
    mel: torch.Tensor
    mask: torch.Tensor
    emb_pitch: torch.Tensor
    emb_content: torch.Tensor
    phonemes: torch.Tensor | None = None
    labels: Label | None = None

    def __len__(self) -> int:
        """Get the number of samples in the batch."""
        return self.batch_size

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
        if self.phonemes is not None:
            self.phonemes = self.phonemes.to(device, non_blocking=True)
        if self.labels is not None:
            self.labels.label_tensor = self.labels.label_tensor.to(
                device, non_blocking=True
            )
        return self


class BasePreprocessor(nn.Module):
    def __init__(
        self,
        mel_transform: MelTransform,
        pitch_encoder: VQVAEEncoder | YINEncoder,
        content_encoder: XLSR | Hubert | XLSR_ESPEAK_CTC,
        sample_rate: int,
        perturb_inputs: bool = False,
        flatten_pitch: bool = False,
    ) -> None:
        super().__init__()
        self.mel_transform = mel_transform
        self.pitch_encoder = pitch_encoder
        self.content_encoder = content_encoder
        self.sample_rate = sample_rate
        self.perturb_inputs = perturb_inputs
        self._praat_processor = PraatProcessor(sample_rate, flatten_pitch)

    def __call__(
        self,
        audio: torch.Tensor,
        n_frames: torch.Tensor | None = None,
    ) -> DDDMInput:
        return super().__call__(audio, n_frames)


class DDDMPreprocessor(BasePreprocessor):
    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        n_frames: torch.Tensor | None = None,
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

        if self.perturb_inputs and self.training:
            detached = audio.detach().cpu()
            audio_p = self._praat_processor.g_batched(detached).to(audio.device)
            audio_c = self._praat_processor.f_batched(detached).to(audio.device)
        else:
            detached = audio.detach()
            audio_p = detached
            audio_c = detached

        emb_pitch = self.pitch_encoder(audio_p)

        # ensure xlsr/hubert embedding and x_mask are aligned
        audio_c = util.pad_for_xlsr(audio_c, self.sample_rate)
        emb_content = self.content_encoder(audio_c)

        return DDDMInput(
            audio=audio,
            mel=mel,
            mask=mask.detach(),
            emb_pitch=emb_pitch.detach(),
            emb_content=emb_content.detach(),
        )


class DurDDDMPreprocessor(BasePreprocessor):
    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        n_frames: torch.Tensor | None = None,
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

        if self.perturb_inputs and self.training:
            detached = audio.detach().cpu()
            audio_p = self._praat_processor.g_batched(detached).to(audio.device)
            audio_c = self._praat_processor.f_batched(detached).to(audio.device)
        else:
            detached = audio.detach()
            audio_p = detached
            audio_c = detached

        emb_pitch = self.pitch_encoder(audio_p)

        # ensure xlsr/hubert embedding and x_mask are aligned
        audio_c = util.pad_for_xlsr(audio_c, self.sample_rate)
        phonemes, emb_content = self.content_encoder(audio_c)

        return DDDMInput(
            audio=audio,
            mel=mel,
            mask=mask.detach(),
            emb_pitch=emb_pitch.detach(),
            emb_content=emb_content.detach(),
            phonemes=phonemes,
        )
