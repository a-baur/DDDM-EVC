import torch
import torch.nn as nn
import torch.nn.functional as F

import util
from config import SrcFtrEncoderConfig
from modules.wavenet_decoder import Decoder

from .speaker_encoder import MetaStyleSpeech


class SourceFilterEncoder(nn.Module):
    def __init__(self, cfg: SrcFtrEncoderConfig):
        super().__init__()
        self.emb_c = nn.Conv1d(1024, cfg.decoder.hidden_dim, 1)
        self.emb_f0 = nn.Embedding(cfg.pitch_encoder.vq.k_bins, cfg.decoder.hidden_dim)
        self.emb_g = MetaStyleSpeech(cfg.speaker_encoder)
        self.dec_f = Decoder(cfg.decoder)
        self.dec_s = Decoder(cfg.decoder)

    def _get_embeddings(
        self,
        x_mel: torch.Tensor,
        length: torch.Tensor,
        conent_enc: torch.Tensor,
        pitch_enc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from the input tensors.

        :param x_mel: Input mel-spectrogram
        :param length: Length of unpadded mel-spectrogram
        :param conent_enc: Content encoder output
        :param pitch_enc: Pitch encoder output
        :return: Content, F0, Mask, and Global embeddings
        """
        content = self.emb_c(conent_enc)

        f0 = self.emb_f0(pitch_enc).transpose(1, 2)
        f0 = F.interpolate(f0, content.shape[-1])

        x_mask = util.sequence_mask(length, x_mel.size(2)).to(x_mel.dtype)
        x_mask = x_mask.unsqueeze(1)

        g = self.emb_g(x_mel, x_mask).unsqueeze(-1)

        return content, f0, x_mask, g

    def forward(
        self,
        x_mel: torch.Tensor,
        length: torch.Tensor,
        content_enc: torch.Tensor,
        pitch_enc: torch.Tensor,
        mixup: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param x_mel: Input mel-spectrogram
        :param length: Length of unpadded mel-spectrogram
        :param content_enc: Content encoder output
        :param pitch_enc: Pitch encoder output
        :param mixup: Whether to use prior mixup or not
        :return: Style, Source, and Filter embeddings
        """
        content, f0, x_mask, g = self._get_embeddings(
            x_mel, length, content_enc, pitch_enc
        )

        if mixup:
            g = torch.cat([g, g[torch.randperm(g.size()[0])]], dim=0)
            content = torch.cat([content, content], dim=0)
            f0 = torch.cat([f0, f0], dim=0)
            x_mask = torch.cat([x_mask, x_mask], dim=0)

        y_ftr = self.dec_f(F.relu(content), x_mask, g=g)
        y_src = self.dec_s(f0, x_mask, g=g)

        return g, y_src, y_ftr

    def voice_conversion(
        self,
        y_mel: torch.Tensor,
        y_length: torch.Tensor,
        x_length: torch.Tensor,
        content_enc: torch.Tensor,
        pitch_enc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Voice conversion using the Source-Filter encoder.

        :param y_mel: Mel-spectrogram of the target speaker
        :param y_length: Length of unpadded target mel-spectrogram
        :param x_length: Length of unpadded source mel-spectrogram
        :param content_enc: Content encoder output
        :param pitch_enc: Pitch encoder output
        :return: Converted mel-spectrogram, Style embedding,
            Source embedding, and Filter embedding
        """
        y_mask = util.sequence_mask(y_length, content_enc.size(2)).to(content_enc.dtype)
        y_mask = y_mask.unsqueeze(1)

        content, f0, x_mask, g = self._get_embeddings(
            y_mel, x_length, content_enc, pitch_enc
        )

        out_ftr = self.dec_f(F.relu(content), y_mask, g=g)
        out_src = self.dec_s(f0, y_mask, g=g)

        out = out_ftr + out_src

        return out, g, out_src, out_ftr
