import torch
import transformers
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from util import forward_fill


class XLSR(torch.nn.Module):
    """
    Wav2Vec2 model for feature extraction

    :param layer: layer to extract features from
    """

    def __init__(self, layer: int = 12) -> None:
        super().__init__()
        self.wav2vec2 = transformers.Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2(x, output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]
        return y.permute((0, 2, 1))


class XLSR_ESPEAK_CTC(torch.nn.Module):
    """
    Wav2Vec2 model for feature extraction

    :param layer: layer to extract features from
    """

    def __init__(
        self,
        return_logits: bool = True,
        return_hidden: bool = False,
        logits_to_phoneme: bool = False,
        layer: int = 12,
    ) -> None:
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-lv-60-espeak-cv-ft"
        )
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )

        self.wav2vec2.requires_grad_(False)
        self.wav2vec2.eval()
        self.feature_layer = layer
        self.return_logits = return_logits
        self.return_hidden = return_hidden
        self.logits_to_phoneme = logits_to_phoneme

    def _logits_to_phoneme_sequence(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logits_to_phoneme:
            return forward_fill(torch.argmax(logits, dim=-1))
        return logits.permute((0, 2, 1))

    @torch.no_grad()  # type: ignore
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = (
            self.processor(
                x,
                return_tensors="pt",
                sampling_rate=16000,
            )
            .input_values.squeeze(0)
            .to(x.device)
        )
        outputs = self.wav2vec2(x, output_hidden_states=self.return_hidden)
        if self.return_logits and self.return_hidden:
            hidden_states = outputs.hidden_states[self.feature_layer].permute((0, 2, 1))
            logits = self._logits_to_phoneme_sequence(outputs.logits)
            return logits, hidden_states
        elif self.return_hidden:
            hidden_states = outputs.hidden_states[self.feature_layer].permute((0, 2, 1))
            return hidden_states
        elif self.return_logits:
            return self._logits_to_phoneme_sequence(outputs.logits)
        else:
            raise ValueError("Model must return either logits or hidden states")


class Hubert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hubert = transformers.HubertModel.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        for param in self.hubert.parameters():
            param.requires_grad = False
            param.grad = None
        self.hubert.eval()

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.hubert(x)
        y = outputs.last_hidden_state
        return y.permute((0, 2, 1))
