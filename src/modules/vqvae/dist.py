# Adapted from https://github.com/openai/jukebox

from enum import Enum

import torch
import torch.distributed as dist


class ReduceOp(Enum):
    SUM = (0,)
    PRODUCT = (1,)
    MIN = (2,)
    MAX = 3

    def to_dist_op(self) -> dist.ReduceOp:
        return {
            self.SUM: dist.ReduceOp.SUM,
            self.PRODUCT: dist.ReduceOp.PRODUCT,
            self.MIN: dist.ReduceOp.MIN,
            self.MAX: dist.ReduceOp.MAX,
        }[self]


def is_available() -> bool:
    return dist.is_initialized()


def get_rank() -> int:
    if is_available():
        return dist.get_rank()
    else:
        return 0


def all_reduce(
    tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM
) -> torch.Tensor | None:
    if is_available():
        return dist.all_reduce(tensor, op.to_dist_op())
    return None


def broadcast(tensor: torch.Tensor, src: int) -> dist.Work | None:
    if is_available():
        return dist.broadcast(tensor, src)
    return None
