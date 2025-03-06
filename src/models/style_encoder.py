import torch
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from config import MetaStyleSpeechConfig, StyleEncoderConfig
from models.dddm.input import DDDMInput
from modules.commons import Mish
from modules.style_speech import Conv1dGLU, MultiHeadAttention
from modules.w2v2_l_robust import RegressionHead
from util import temporal_avg_pool


class StyleEncoder(nn.Module):
    def __init__(self, cfg: StyleEncoderConfig):
        super().__init__()
        self.speaker_encoder = MetaStyleSpeech(cfg.speaker_encoder)
        self.emotion_encoder = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME)

        spk_dim = cfg.speaker_encoder.out_dim
        emo_dim = cfg.emotion_emb_dim
        cond_total = spk_dim + emo_dim
        self.emotion_emb = nn.Linear(cfg.emotion_encoder.out_dim, emo_dim)
        self.cond_acts = nn.Sequential(
            nn.Linear(cond_total, cond_total // 2),
            nn.ReLU(),
            nn.Linear(cond_total // 2, cond_total),
            nn.Sigmoid(),
        )

    def forward(self, x: DDDMInput) -> torch.Tensor:
        """
        Encode condition tensor.

        :param x: DDDM input object
        :return: Condition tensor
        """
        emo = self.emotion_encoder(x.audio, embeddings_only=True)
        emo = self.emotion_emb(emo)
        spk = self.speaker_encoder(x)
        cond = torch.cat([spk, emo], dim=1)
        cond = cond * self.cond_acts(cond)
        return cond


class W2V2LRobust(Wav2Vec2PreTrainedModel):
    """Speech emotion classifier."""

    MODEL_NAME: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__(config)

        # Load pretrained components
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

        self.processor = Wav2Vec2Processor.from_pretrained(self.MODEL_NAME)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        embeddings_only: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param x: input tensor
        :param embeddings_only: If true, only return hidden states.
        :return: Hidden states and logits
        """
        np_batch = [s.cpu().detach().numpy() for s in x.unbind(0)]
        x_norm: torch.Tensor = self.processor(
            np_batch,
            return_tensors="pt",
            sampling_rate=16000,
            return_attention_mask=False,
        )["input_values"].to(x.device)

        outputs = self.wav2vec2(x_norm)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)

        if embeddings_only:
            return hidden_states

        logits = self.classifier(hidden_states)
        return hidden_states, logits


class MetaStyleSpeech(nn.Module):
    """
    Style encoder module.

    Based one Mel-Style Encoder from Meta-StyleSpeech.

    Description:

    1.) Spectral Processing:
        Embedding the input mel-spectrogram
        into a hidden representation.

    2.) Temporal Processing:
        Capture sequential information
        from the hidden representation.

    3.) Multi-Head Self-Attention:
        Capture global information from
        the hidden representation.

    4.) Temporal Average Pooling:
        Aggregate the hidden representation
         into a single vector.

    Reference:
    Meta-StyleSpeech: Multi-Speaker Adaptive Text-to-Speech Generation
    https://arxiv.org/abs/2004.04634
    """

    def __init__(self, cfg: MetaStyleSpeechConfig) -> None:
        super().__init__()

        self.in_dim = cfg.in_dim
        self.hidden_dim = cfg.hidden_dim
        self.out_dim = cfg.out_dim
        self.kernel_size = 5
        self.n_head = 2
        self.dropout = 0.1

        self.spectral = nn.Sequential(
            nn.Conv1d(self.in_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
            Mish(),
            nn.Dropout(self.dropout),
        )

        self.temporal = nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(
            self.hidden_dim,
            self.hidden_dim,
            self.n_head,
            p_dropout=self.dropout,
            proximal_bias=False,
            proximal_init=True,
        )

        self.atten_drop = nn.Dropout(self.dropout)
        self.fc = nn.Conv1d(self.hidden_dim, self.out_dim, 1)

    def forward(self, x: DDDMInput) -> torch.Tensor:
        _x = self.spectral(x.mel) * x.mask
        _x = self.temporal(_x) * x.mask

        # Self-attention mask to prevent
        # attending to padding tokens.
        # mask = (B, 1, T)
        # attn_mask = (B, 1, 1, T) * (B, 1, T, 1) -> (B, 1, T, T)
        attn_mask = x.mask.unsqueeze(2) * x.mask.unsqueeze(-1)
        y = self.slf_attn(_x, _x, attn_mask=attn_mask)
        _x = _x + self.atten_drop(y)

        _x = self.fc(_x)

        w = temporal_avg_pool(_x, mask=x.mask)

        return w
