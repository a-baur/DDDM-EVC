import torch
import torch.nn as nn

import util
from config import ModelConfig
from models.content_encoder import Wav2Vec2
from models.pitch_encoder import VQVAEEncoder
from models.wavenet_decoder import WavenetDecoder


class SourceFilterEncoder(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        sample_rate: int,
        content_encoder: Wav2Vec2 = None,
        pitch_encoder: VQVAEEncoder = None,
        decoder: WavenetDecoder = None,
    ) -> None:
        super().__init__()
        self.content_encoder = content_encoder or Wav2Vec2()
        self.pitch_encoder = pitch_encoder or VQVAEEncoder(cfg.pitch_encoder)
        self.decoder = decoder or WavenetDecoder(cfg)
        self.sample_rate = sample_rate

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        mixup_ratios: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param x: Input waveform
        :param x_mask: Mask for the input mel-spectrogram
        :param g: Global conditioning tensor
        :param mixup_ratios: Ratio of the mixup interpolation
        :return: Source, and Filter representations
        """
        f0 = util.get_normalized_f0(x, self.sample_rate)

        # ensure xlsr embedding and x_mask are aligned
        x_pad = util.pad_audio_for_xlsr(x)

        x_emb_content = self.content_encoder(x_pad)
        x_emb_pitch = self.pitch_encoder(f0)

        src_mel, ftr_mel = self.decoder(
            x_emb_content, x_emb_pitch, g, x_mask, mixup_ratios=mixup_ratios
        )

        return src_mel, ftr_mel
