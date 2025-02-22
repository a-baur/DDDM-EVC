import torch
from torch import nn as nn
from torch.nn import functional as F

import util
from config import DDDMEVCConfig, DDDMVCConfig, WavenetDecoderConfig
from models.content_encoder import XLSR
from models.pitch_encoder import VQVAEEncoder
from modules.wavenet_decoder import Decoder


class WavenetDecoder(nn.Module):
    def __init__(self, cfg: WavenetDecoderConfig, content_dim: int, f0_dim: int):
        super().__init__()
        self.emb_c = nn.Conv1d(content_dim, cfg.hidden_dim, 1)
        self.emb_f0 = nn.Embedding(f0_dim, cfg.hidden_dim)
        self.dec_ftr = Decoder(cfg)
        self.dec_src = Decoder(cfg)

    def forward(
        self,
        content_enc: torch.Tensor,
        pitch_enc: torch.Tensor,
        g: torch.Tensor,
        mask: torch.Tensor,
        mixup_ratios: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param content_enc: Content encoder output
        :param pitch_enc: Pitch encoder output
        :param g: Global conditioning tensor
        :param mask: Mask for the input mel-spectrogram
        :param mixup_ratios: Mixup ratios for each sample in the batch (B,)
        :return: Source, and Filter representations
        """
        content = self.emb_c(content_enc)
        f0 = self.emb_f0(pitch_enc).transpose(1, 2)
        f0 = F.interpolate(f0, content.shape[-1])  # match the length of content emb

        if mixup_ratios is not None:
            batch_size = content_enc.size(0)
            # Randomly shuffle the speaker embeddings
            random_style = g[torch.randperm(g.size()[0])]

            # Concatenate mixed up batches to the original batch
            g = torch.cat([g, random_style], dim=0)
            content = torch.cat([content, content], dim=0)
            f0 = torch.cat([f0, f0], dim=0)
            mask = torch.cat([mask, mask], dim=0)

            # Decode the source and filter representations
            y_ftr = self.dec_ftr(F.relu(content), mask, g=g)
            y_src = self.dec_src(f0, mask, g=g)

            # Mixup the outputs according to the mixup ratio
            mixup_ratios = mixup_ratios[:, None, None]  # (B) -> (B x 1 x 1)
            y_src = (
                mixup_ratios * y_src[:batch_size, :, :]
                + (1 - mixup_ratios) * y_src[batch_size:, :, :]
            )
            y_ftr = (
                mixup_ratios * y_ftr[:batch_size, :, :]
                + (1 - mixup_ratios) * y_ftr[batch_size:, :, :]
            )
        else:
            y_ftr = self.dec_ftr(F.relu(content), mask, g=g)
            y_src = self.dec_src(f0, mask, g=g)

        return y_src, y_ftr


class SourceFilterEncoder(nn.Module):
    def __init__(
        self,
        cfg: DDDMVCConfig | DDDMEVCConfig,
        sample_rate: int,
        content_encoder: XLSR = None,
        pitch_encoder: VQVAEEncoder = None,
        decoder: WavenetDecoder = None,
    ) -> None:
        super().__init__()
        self.content_encoder = content_encoder or XLSR()
        self.pitch_encoder = pitch_encoder or VQVAEEncoder(cfg.pitch_encoder)
        self.decoder = decoder or WavenetDecoder(
            cfg.decoder,
            content_dim=cfg.content_encoder.out_dim,
            f0_dim=cfg.pitch_encoder.vq.k_bins,
        )
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
        x_pad = util.pad_audio_for_xlsr(x, self.sample_rate)

        x_emb_content = self.content_encoder(x_pad)
        x_emb_pitch = self.pitch_encoder(f0)

        src_mel, ftr_mel = self.decoder(
            x_emb_content, x_emb_pitch, g, x_mask, mixup_ratios=mixup_ratios
        )

        return src_mel, ftr_mel
