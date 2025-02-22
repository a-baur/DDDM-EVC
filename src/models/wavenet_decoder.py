import torch
import torch.nn as nn
import torch.nn.functional as F

from config import WavenetDecoderConfig
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
