import pathlib
from typing import Iterable, Literal, Optional

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
    mode: Literal["train", "eval"] = "eval",
) -> None:
    """Load model from checkpoint file."""
    ckpt_dir = util.get_root_path() / "ckpt"
    ckpt = torch.load(ckpt_dir / ckpt_file, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    if freeze:
        model.requires_grad_(False)
    if mode == "eval":
        model.eval()
    else:
        model.train()


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


def clip_grad_value(
    parameters: Iterable[torch.Tensor],
    clip_value: Optional[float],
    norm_type: float = 2.0,
) -> float:
    """
    Clips the gradient values of the given parameters
    to a specified range and computes the total norm.

    :param parameters: Iterable of model parameters whose
        gradients will be clipped.
    :param clip_value: Maximum absolute value for gradient
        clipping. If None, no clipping is applied.
    :param norm_type: Type of norm to compute total gradient
        norm. Default is 2 (Euclidean norm).
    :return: The total norm of the gradients before clipping.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            if clip_value is not None:
                p.grad.data.clamp_(min=-clip_value, max=clip_value)

    return total_norm ** (1.0 / norm_type)
