import torch
import transformers


class XLSR(torch.nn.Module):
    """
    Wav2Vec2 model for feature extraction

    :param layer: layer to extract features from
    """

    def __init__(self, layer: int = 12) -> None:
        super().__init__()
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained(
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


class Hubert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hubert = transformers.HubertPreTrainedModel.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        for param in self.hubert.parameters():
            param.requires_grad = False
            param.grad = None
        self.hubert.eval()

    @torch.no_grad()  # type: ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hubert(x)
