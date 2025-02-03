from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch


def plot_mel(
    mel: torch.Tensor,
    title: str = "Mel-spectrogram",
    save_path: Optional[Path] = None,
) -> None:
    """Plot mel-spectrogram."""
    plt.figure(figsize=(10, 5))
    plt.imshow(mel[0].cpu().detach().numpy(), aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
