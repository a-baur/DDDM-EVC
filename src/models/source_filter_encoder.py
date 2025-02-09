import torch
import torch.nn as nn

import util
from config import ModelsConfig
from models.content_encoder import Wav2Vec2
from models.pitch_encoder import VQVAE
from models.wavenet_decoder import WavenetDecoder


class SourceFilterEncoder(nn.Module):
    def __init__(
        self,
        cfg: ModelsConfig,
        content_encoder: Wav2Vec2 = None,
        pitch_encoder: VQVAE = None,
        decoder: WavenetDecoder = None,
    ) -> None:
        super().__init__()
        self.content_encoder = content_encoder or Wav2Vec2()
        self.pitch_encoder = pitch_encoder or VQVAE(cfg.pitch_encoder)
        self.decoder = decoder or WavenetDecoder(cfg)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor,
        mixup: bool = False,
        sample_rate: int = 16000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param x: Input waveform
        :param x_mask: Mask for the input mel-spectrogram
        :param g: Global conditioning tensor
        :param mixup: Whether to use prior mixup or not
        :param sample_rate: Sampling rate of the input waveform
        :return: Source, and Filter representations
        """
        f0 = util.get_normalized_f0(x, sample_rate)

        x_emb_content = self.content_encoder(x)
        x_emb_pitch = self.pitch_encoder.code_extraction(f0)

        src_mel, ftr_mel = self.decoder(
            x_emb_content, x_emb_pitch, g, x_mask, mixup=mixup
        )

        return src_mel, ftr_mel
