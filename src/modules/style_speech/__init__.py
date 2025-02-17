"""
Speaker encoder module, based on the
Mel-Style Encoder from Meta-StyleSpeech.

Implementation based on
DDDM-VC: https://github.com/hayeong0/DDDM-VC/blob/7f826a366b2941c7f020de07956bf5161c4979b4/model/styleencoder.py
StyleSpeech: https://github.com/KevinMIN95/StyleSpeech/blob/f939cf9cb981db7b738fa9c9c9a7fea2dfdd0766/models/StyleSpeech.py#L251
"""  # noqa: E501

from .modules import Conv1dGLU, MultiHeadAttention

__all__ = [
    "MultiHeadAttention",
    "Conv1dGLU",
]
