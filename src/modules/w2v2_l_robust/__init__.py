"""
Wav2Vec2 Large Robust SER model.

Paper: https://zenodo.org/records/6221127

Based on implementation from
https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
"""

from .modules import RegressionHead

__all__ = ["RegressionHead"]
