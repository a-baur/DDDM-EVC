import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Processor,
    WavLMModel,
)
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
from speechbrain.inference.speaker import EncoderClassifier
from huggingface_hub import hf_hub_download
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForAudioClassification

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
            emo_path = (
                get_root_path() / "avgclass_emo_embeds" / "emo_wav2vec2" / emo_dim
            )
            emo_avg = torch.load(emo_path / f"{emo_level}.pt").to(x.audio.device)
            emo_avg = emo_avg.unsqueeze(0).expand(x.batch_size, -1)
            emo = emo * (1 - emo_factor) + emo_avg * emo_factor
        if spk_factor > 0:
            spk_path = (
                get_root_path()
                / "avgclass_emo_embeds"
                / "spk_metastylespeech"
                / emo_dim
            )
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
        self.speaker_encoder = MetaStyleSpeech()
        self.emotion_encoder = W2V2LRobust.from_pretrained(W2V2LRobust.MODEL_NAME)

        self.speaker_encoder.eval().requires_grad_(False)
        self.emotion_encoder.eval().requires_grad_(False)

        self.emo_proj = nn.Linear(1024, hidden_dim)
        self.spk_proj = nn.Linear(256, hidden_dim)

        self.l2_normalize = cfg.l2_normalize

    def encode(
        self,
        x: DDDMInput,
        emo_level: int = None,
        emo_dim: str = "EmoAct",
        emo_factor: float = 0.0,
        spk_factor: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emo = self.emotion_encoder(x.audio)
        spk = self.speaker_encoder(x.audio)

        if emo_level and emo_factor > 0:
            emo_path = (
                get_root_path() / "avgclass_emo_embeds" / "emo_wav2vec2" / emo_dim
            )
            emo_avg = torch.load(emo_path / f"{emo_level}.pt").to(x.audio.device)
            emo_avg = emo_avg.unsqueeze(0).expand(x.batch_size, -1)
            emo = emo * (1 - emo_factor) + emo_avg * emo_factor
        if emo_level and spk_factor > 0:
            spk_path = get_root_path() / "avgclass_emo_embeds" / "spk_ecapa" / emo_dim
            spk_avg = torch.load(spk_path / f"{emo_level}.pt").to(x.audio.device)
            spk_avg = spk_avg.unsqueeze(0).expand(x.batch_size, -1)
            spk = spk * (1 - spk_factor) + spk_avg * spk_factor

        emo = self.emo_proj(emo)
        spk = self.spk_proj(spk)

        if self.l2_normalize:
            emo = F.normalize(emo, p=2, dim=1)
            spk = F.normalize(spk, p=2, dim=1)

        return spk, emo

    @torch.no_grad()
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


class WavLM_Odyssey(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForAudioClassification.from_pretrained(
            "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
            trust_remote_code=True,
        )
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[0], x.shape[1]).to(x.device)
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        x = self.model.ssl_model(x, attention_mask=attn_mask).last_hidden_state
        x = self.model.pool_model(x, mask)
        return x


class WavLM_Large(nn.Module):
    def __init__(self):
        super().__init__()

        model_id = "microsoft/wavlm-large"

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = WavLMModel.from_pretrained(model_id)

        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_device = x.device
        inputs = self.feature_extractor(
            x.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        ).to(original_device)
        x = self.model(**inputs).last_hidden_state
        return x.mean(dim=1)  # Average pooling over time dimension


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
        embeddings_only: bool = True,
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


class ECAPA_TDNN(nn.Module):
    def __init__(self):
        super().__init__()
        save_dir = get_root_path() / "pretrained" / "ecapa_tdnn"
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cuda"},
            savedir=save_dir.as_posix(),
            huggingface_cache_dir=os.environ.get("HF_HOME"),
        )
        self.model.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_batch(x, normalize=True).squeeze(1)


class ECAPA2(nn.Module):
    def __init__(self):
        super().__init__()
        model_file = hf_hub_download(repo_id="Jenthe/ECAPA2", filename="ecapa2.pt")
        self.model = torch.jit.load(
            model_file, map_location="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.half()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.jit.optimized_execution(False):
            out = self.model(x).squeeze(1)
        return out


class ECAPA_TDNN_NEMO(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name="ecapa_tdnn"
        )
        self.model.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.int64).to(x.device)
        return self.model(x, lengths)[1]
