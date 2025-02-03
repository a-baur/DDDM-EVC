import pathlib

import torch

import util


def get_root_path() -> pathlib.Path:
    """Returns the root directory of the project"""
    current_dir = pathlib.Path(__file__).parent
    while not (current_dir / "pyproject.toml").exists():
        current_dir = current_dir.parent
    return current_dir


def load_model(model: torch.nn.Module, ckpt_file: str | pathlib.Path) -> None:
    """Load model from checkpoint file."""
    ckpt_dir = util.get_root_path() / "ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_dir / ckpt_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
