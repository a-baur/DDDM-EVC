"""
Attention-based modules.

Based on the implementation from
DDDM-VC: https://github.com/hayeong0/DDDM-VC/blob/7f826a366b2941c7f020de07956bf5161c4979b4/modules_sf/attentions.py
Glow-TTS: https://github.com/jaywalnut310/glow-tts/blob/13e997689d643410f5d9f1f9a73877ae85e19bc2/attentions.py#L106

Modifications:
- moved static helper functions outside the class
"""  # noqa: E501

from .modules import Conv1dGLU, MultiHeadAttention

__all__ = [
    "MultiHeadAttention",
    "Conv1dGLU",
]
