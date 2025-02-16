import os
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


def get_cuda_devices() -> list[str]:
    """Get list of cuda devices available for training."""
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    n_gpus = torch.cuda.device_count()
    if cuda_visible_devices is not None:
        gpu_ids = list(map(int, cuda_visible_devices.split(",")))
    else:
        gpu_ids = list(range(n_gpus))

    device_info = []
    for idx, i in zip(gpu_ids, range(n_gpus)):
        name = torch.cuda.get_device_name(i)
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        mem_free, mem_total = mem_free / 1024**3, mem_total / 1024**3
        mem_usage = mem_total - mem_free
        percent = mem_usage / mem_total
        info = f"[gpu:{idx}, cuda:{i}] {name} (Utilization: {percent:7.2%} [{mem_usage:4.1f}GB/{mem_total:4.1f}GB])"
        device_info.append(info)

    return device_info
