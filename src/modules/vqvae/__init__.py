"""
VQ-VAE implementation.

Based on the implementation from
DDDM-VC: https://github.com/hayeong0/DDDM-VC/tree/master/modules_vqvae
OpenAI Jukebox: https://github.com/openai/jukebox
"""

from .jukebox import Decoder, Encoder
from .vq import Bottleneck

__all__ = ["Encoder", "Decoder", "Bottleneck"]
