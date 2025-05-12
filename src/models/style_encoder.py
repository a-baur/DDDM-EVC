import torch
import torch.nn.functional as F
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from config import MetaStyleSpeechConfig, StyleEncoderConfig
from models.dddm.preprocessor import DDDMInput
from modules.commons import Mish
from modules.style_speech import Conv1dGLU, MultiHeadAttention
from modules.w2v2_l_robust import RegressionHead
from util import get_root_path, temporal_avg_pool


def CCCLoss(preds, labels, eps=1e-8):
    """
    Computes Concordance Correlation Coefficient loss.

    :param preds: [batch_size, dims]
    :param labels: [batch_size, dims]
    :return: loss = 1 - mean(CCC over dims)
    """
    preds_mean = preds.mean(dim=0)
    labels_mean = labels.mean(dim=0)

    covariance = ((preds - preds_mean) * (labels - labels_mean)).mean(dim=0)
    preds_var = preds.var(dim=0)
    labels_var = labels.var(dim=0)

    ccc = (2 * covariance) / (
        preds_var + labels_var + (preds_mean - labels_mean).pow(2) + eps
    )

    return 1 - ccc.mean()


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


class StyleLabelEncoder(nn.Module):
    def __init__(self, cfg: StyleEncoderConfig):
        super().__init__()
        self.speaker_encoder = MetaStyleSpeech(cfg.speaker_encoder)

        self.speaker_encoder.requires_grad_(False)

        emo_dim = cfg.emotion_emb_dim
        self.emotion_emb = nn.Linear(cfg.emotion_encoder.out_dim, emo_dim)

        self.l2_normalize = cfg.l2_normalize
        self.p_spk_masking = cfg.p_spk_masking
        self.p_emo_masking = cfg.p_emo_masking

    def emotion_conversion(self, x: DDDMInput, emo_level: int) -> torch.Tensor:
        assert x.label is not None, "Label is None. Cannot encode condition tensor."

        emo = x.label.label_tensor[:, 0:3]
        emo[:, 0] = emo_level
        emo = self.emotion_emb(emo)
        spk = self.speaker_encoder(x)

        if self.l2_normalize:
            spk = F.normalize(spk, p=2, dim=1)
            emo = F.normalize(emo, p=2, dim=1)

        cond = torch.cat([spk, emo], dim=1)
        return cond

    def forward(self, x: DDDMInput) -> torch.Tensor:
        """
        Encode condition tensor.

        :param x: DDDM input object
        :return: Condition tensor
        """
        assert x.label is not None, "Label is None. Cannot encode condition tensor."

        emo = x.label.label_tensor[:, 0:3]
        emo = self.emotion_emb(emo)
        spk = self.speaker_encoder(x)

        if self.l2_normalize:
            spk = F.normalize(spk, p=2, dim=1)
            emo = F.normalize(emo, p=2, dim=1)

        if self.p_emo_masking > 0 and self.training:
            r = torch.rand(x.batch_size, 1).to(x.audio.device)
            emo_mask = 1 - (r < self.p_emo_masking).float()
            emo = emo * emo_mask

        if self.p_spk_masking > 0 and self.training:
            r = torch.rand(x.batch_size, 1).to(x.audio.device)
            spk_mask = 1 - (r < self.p_spk_masking).float()
            spk = spk * spk_mask

        cond = torch.cat([spk, emo], dim=1)
        return cond


class StyleEncoder(nn.Module):
    def __init__(self, cfg: StyleEncoderConfig):
        super().__init__()
        self.speaker_encoder = MetaStyleSpeech(cfg.speaker_encoder)
        self.emotion_encoder = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME)

        self.speaker_encoder.requires_grad_(False)
        self.emotion_encoder.requires_grad_(False)

        emo_dim = cfg.emotion_emb_dim
        self.emotion_emb = nn.Linear(cfg.emotion_encoder.out_dim, emo_dim)

        self.l2_normalize = cfg.l2_normalize
        self.p_spk_masking = cfg.p_spk_masking
        self.p_emo_masking = cfg.p_emo_masking

    def emotion_conversion(
        self,
        x: DDDMInput,
        emo_level: int,
        emo_dim: str = "EmoAct",
        emo_factor: float = 0.0,
        spk_factor: float = 0.0,
    ) -> torch.Tensor:
        emo = self.emotion_encoder(x.audio, embeddings_only=True)
        spk = self.speaker_encoder(x)

        if emo_factor > 0:
            emo_path = get_root_path() / "avgclass_emo_embeds" / "emo" / emo_dim
            emo_avg = torch.load(emo_path / f"{emo_level}.pt").to(x.audio.device)
            emo_avg = emo_avg.unsqueeze(0).expand(x.batch_size, -1)
            emo = emo * (1 - emo_factor) + emo_avg * emo_factor
        if spk_factor > 0:
            spk_path = get_root_path() / "avgclass_emo_embeds" / "spk" / emo_dim
            spk_avg = torch.load(spk_path / f"{emo_level}.pt").to(x.audio.device)
            spk_avg = spk_avg.unsqueeze(0).expand(x.batch_size, -1)
            spk = spk * (1 - spk_factor) + spk_avg * spk_factor

        emo = self.emotion_emb(emo)

        if self.l2_normalize:
            spk = F.normalize(spk, p=2, dim=1)
            emo = F.normalize(emo, p=2, dim=1)

        cond = torch.cat([spk, emo], dim=1)
        return cond

    def forward(self, x: DDDMInput) -> torch.Tensor:
        """
        Encode condition tensor.

        :param x: DDDM input object
        :return: Condition tensor
        """
        emo = self.emotion_encoder(x.audio, embeddings_only=True)
        emo = self.emotion_emb(emo)
        spk = self.speaker_encoder(x)

        if self.l2_normalize:
            spk = F.normalize(spk, p=2, dim=1)
            emo = F.normalize(emo, p=2, dim=1)

        if self.p_emo_masking > 0 and self.training:
            r = torch.rand(x.batch_size, 1).to(x.audio.device)
            emo_mask = 1 - (r < self.p_emo_masking).float()
            emo = emo * emo_mask

        if self.p_spk_masking > 0 and self.training:
            r = torch.rand(x.batch_size, 1).to(x.audio.device)
            spk_mask = 1 - (r < self.p_spk_masking).float()
            spk = spk * spk_mask

        cond = torch.cat([spk, emo], dim=1)
        return cond


class DisentangledStyleEncoder(nn.Module):
    def __init__(self, cfg: StyleEncoderConfig, hidden_dim: int = 256, n_spk: int = 10):
        super().__init__()
        self.speaker_encoder = MetaStyleSpeech(cfg.speaker_encoder)
        self.emotion_encoder = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME)

        self.speaker_encoder.eval().requires_grad_(False)
        self.emotion_encoder.eval().requires_grad_(False)

        self.emo_proj = nn.Linear(cfg.emotion_encoder.out_dim, hidden_dim, bias=False)
        self.spk_proj = nn.Sequential(
            nn.Linear(cfg.speaker_encoder.out_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.n_spk = n_spk + 1  # +1 for unknown speaker

        self.emo_reg = nn.Linear(hidden_dim, 3)
        self.spk_cls = nn.Linear(hidden_dim, self.n_spk)

        self.emo_adv = nn.Linear(hidden_dim, self.n_spk)
        self.spk_adv = nn.Linear(hidden_dim, 3)

        self.l2_normalize = cfg.l2_normalize

    def encode(
        self,
        x: DDDMInput,
        emo_level: int = None,
        emo_dim: str = "EmoAct",
        emo_factor: float = 0.0,
        spk_factor: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emo = self.emotion_encoder(x.audio, embeddings_only=True)
        spk = self.speaker_encoder(x)

        if emo_level and emo_factor > 0:
            emo_path = get_root_path() / "avgclass_emo_embeds" / "emo" / emo_dim
            emo_avg = torch.load(emo_path / f"{emo_level}.pt").to(x.audio.device)
            emo_avg = emo_avg.unsqueeze(0).expand(x.batch_size, -1)
            emo = emo * (1 - emo_factor) + emo_avg * emo_factor
        if emo_level and spk_factor > 0:
            spk_path = get_root_path() / "avgclass_emo_embeds" / "spk" / emo_dim
            spk_avg = torch.load(spk_path / f"{emo_level}.pt").to(x.audio.device)
            spk_avg = spk_avg.unsqueeze(0).expand(x.batch_size, -1)
            spk = spk * (1 - spk_factor) + spk_avg * spk_factor

        emo = self.emo_proj(emo)
        spk = self.spk_proj(spk)

        if self.l2_normalize:
            emo = F.normalize(emo, p=2, dim=1)
            spk = F.normalize(spk, p=2, dim=1)

        return spk, emo

    def forward(
        self,
        x: DDDMInput,
        emo_level: int = None,
        emo_dim: str = "EmoAct",
        emo_factor: float = 0.0,
        spk_factor: float = 0.0,
    ) -> torch.Tensor:
        """
        Encode condition tensor.

        :param x: DDDM input object
        :param emo_level: Emotion level
        :param emo_dim: Emotion dimension
        :param emo_factor: Emotion factor
        :param spk_factor: Speaker factor
        :return: Condition tensor
        """
        spk, emo = self.encode(x, emo_level, emo_dim, emo_factor, spk_factor)
        return torch.cat([spk, emo], dim=1)

    def compute_loss(
        self,
        x: DDDMInput,
        adv_spk_coef: float = 0.1,
        adv_emo_coef: float = 0.1,
        include_acc: bool = False,
    ):
        spk_target = x.label.spk_id
        known_mask = spk_target >= 0

        assert spk_target.max() <= self.n_spk, (
            f"Speaker ID {spk_target.max()} exceeds number of speakers {self.n_spk}."
        )

        emo_target = x.label.label_tensor[:, 0:3]

        spk, emo = self.encode(x)

        loss_spk = F.cross_entropy(
            self.spk_cls(spk)[known_mask], spk_target[known_mask].long()
        )
        loss_emo = CCCLoss(self.emo_reg(emo), emo_target)

        loss_spk_adv = CCCLoss(self.spk_adv(grad_reverse(spk)), emo_target)
        loss_emo_adv = F.cross_entropy(
            self.emo_adv(grad_reverse(emo))[known_mask], spk_target[known_mask].long()
        )
        
        if include_acc:   
            logits = self.spk_cls(spk)
            acc = (logits.argmax(dim=1) == spk_target.long()).float().mean()
            print(f"Sanity check accuracy: {acc.item()*100:.2f}%")

        loss = (
            loss_spk
            + loss_emo
            + adv_spk_coef * loss_spk_adv
            + adv_emo_coef * loss_emo_adv
        )
        return loss, loss_spk, loss_emo, loss_spk_adv, loss_emo_adv


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
        x: torch.Tensor = self.processor(
            x.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_values.to(x.device)

        outputs = self.wav2vec2(x)
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

    @torch.no_grad()
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
