import torch
import torch.nn as nn

import util
from config import SourceFilterEncoderConfig
from models.content_encoder import Wav2Vec2
from models.pitch_encoder import VQVAE
from models.speaker_encoder import MetaStyleSpeech
from models.wavenet_decoder import WavenetDecoder


class SourceFilterEncoder(nn.Module):
    def __init__(
        self,
        cfg: SourceFilterEncoderConfig,
        content_encoder: Wav2Vec2 = None,
        pitch_encoder: VQVAE = None,
        speaker_encoder: MetaStyleSpeech = None,
        decoder: WavenetDecoder = None,
    ) -> None:
        super().__init__()
        self.content_encoder = content_encoder or Wav2Vec2()
        self.pitch_encoder = pitch_encoder or VQVAE(cfg.pitch_encoder)
        self.speaker_encoder = speaker_encoder or MetaStyleSpeech(cfg.speaker_encoder)
        self.decoder = decoder or WavenetDecoder(cfg)

    def forward(
        self,
        x: torch.Tensor,
        x_mel: torch.Tensor,
        x_mask: torch.Tensor,
        mixup: bool = False,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param x: Input waveform
        :param x_mel: Input mel-spectrogram
        :param x_mask: Mask for the input mel-spectrogram
        :param mixup: Whether to use prior mixup or not
        :param sample_rate: Sampling rate of the input waveform
        :return: Source, and Filter representations
        """
        f0 = util.get_normalized_f0(x, sample_rate)

        x_emb_content = self.content_encoder(x)
        x_emb_pitch = self.pitch_encoder.code_extraction(f0)
        x_emb_spk = self.speaker_encoder(x_mel, x_mask).unsqueeze(-1)

        src_mel, ftr_mel = self.decoder(
            x_emb_spk, x_emb_content, x_emb_pitch, x_mask, mixup=mixup
        )

        return src_mel, ftr_mel

    def voice_conversion(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y_mel: torch.Tensor,
        y_mask: torch.Tensor,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Voice conversion using the Source-Filter encoder.

        :param x: Source waveform
        :param x_mask: Mask for the source mel-spectrogram
        :param y_mel: Target mel-spectrogram
        :param y_mask: Mask for the target mel-spectrogram
        :param sample_rate: Sampling rate of the waveform
        :return: Source, and Filter representations
        """
        f0 = util.get_normalized_f0(x, sample_rate)

        x_emb_content = self.content_encoder(x)
        x_emb_pitch = self.pitch_encoder.code_extraction(f0)
        y_emb_spk = self.speaker_encoder(y_mel, y_mask).unsqueeze(-1)

        src_mel, ftr_mel = self.decoder(
            y_emb_spk, x_emb_content, x_emb_pitch, x_mask, mixup=False
        )

        return src_mel, ftr_mel
