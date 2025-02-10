import pathlib

import torch
from torch import nn

import util


def get_root_path() -> pathlib.Path:
    """Returns the root directory of the project"""
    current_dir = pathlib.Path(__file__).parent
    while not (current_dir / "pyproject.toml").exists():
        current_dir = current_dir.parent
    return current_dir


def load_model(
    model: torch.nn.Module,
    ckpt_file: str | pathlib.Path,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    freeze: bool = False,
) -> None:
    """Load model from checkpoint file."""
    ckpt_dir = util.get_root_path() / "ckpt"
    ckpt = torch.load(ckpt_dir / ckpt_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    if freeze:
        model.requires_grad_(False)
    model.eval()


def init_weights(module: torch.nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    Initialize weights of a module with normal distribution.

    :param module: Module to initialize
    :param mean: Mean of the normal distribution
    :param std: Standard deviation of the normal distribution
    :return: None
    """
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def move_to_device(
    objs: tuple[torch.Tensor | nn.Module, ...], device: torch.device
) -> tuple[torch.Tensor | nn.Module, ...]:
    return tuple(o.to(device) for o in objs)
