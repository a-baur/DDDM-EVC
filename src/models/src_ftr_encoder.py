import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SrcFtrEncoderConfig
from modules.wavenet_decoder import Decoder


class SourceFilterEncoder(nn.Module):
    def __init__(self, cfg: SrcFtrEncoderConfig):
        super().__init__()
        self.emb_c = nn.Conv1d(1024, cfg.decoder.hidden_dim, 1)
        self.emb_f0 = nn.Embedding(cfg.pitch_encoder.vq.k_bins, cfg.decoder.hidden_dim)
        self.dec_ftr = Decoder(cfg.decoder)
        self.dec_src = Decoder(cfg.decoder)

    def forward(
        self,
        speaker_enc: torch.Tensor,
        content_enc: torch.Tensor,
        pitch_enc: torch.Tensor,
        mask: torch.Tensor,
        mixup: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Source-Filter encoder.

        :param speaker_enc: Speaker encoder output
        :param content_enc: Content encoder output
        :param pitch_enc: Pitch encoder output
        :param mask: Mask for the input mel-spectrogram
        :param mixup: Whether to use prior mixup or not
        :return: Source, and Filter representations
        """
        content = self.emb_c(content_enc)
        f0 = self.emb_f0(pitch_enc).transpose(1, 2)
        f0 = F.interpolate(f0, content.shape[-1])  # match the length of content

        if mixup:
            speaker_enc = torch.cat(
                [speaker_enc, speaker_enc[torch.randperm(speaker_enc.size()[0])]], dim=0
            )
            content = torch.cat([content, content], dim=0)
            f0 = torch.cat([f0, f0], dim=0)
            mask = torch.cat([mask, mask], dim=0)

        y_ftr = self.dec_ftr(F.relu(content), mask, g=speaker_enc)
        y_src = self.dec_src(f0, mask, g=speaker_enc)

        return y_src, y_ftr
