import math

import torch
from einops import rearrange

from modules.commons import Mish


class Upsample(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Downsample(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Rezero(torch.nn.Module):
    def __init__(self, fn: torch.nn.Module):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) * self.g


class Block(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=x.dtype)
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(torch.nn.Module):
    def __init__(
        self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8
    ) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv2d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(torch.nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.matmul(k, v.transpose(-1, -2))
        out = torch.matmul(context.transpose(-1, -2), q)
        # context = torch.einsum("bhdn,bhen->bhde", k, v)
        # out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


class Residual(torch.nn.Module):
    def __init__(self, fn: torch.nn.Module) -> None:
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = 1000.0 * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RefBlock(torch.nn.Module):
    def __init__(self, out_dim: int, time_emb_dim: int) -> None:
        super(RefBlock, self).__init__()
        base_dim = out_dim // 4
        self.mlp1 = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, base_dim))
        self.mlp2 = torch.nn.Sequential(
            Mish(), torch.nn.Linear(time_emb_dim, 2 * base_dim)
        )
        self.block11 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2 * base_dim, 3, 1, 1),
            torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
            torch.nn.GLU(dim=1),
        )
        self.block12 = torch.nn.Sequential(
            torch.nn.Conv2d(base_dim, 2 * base_dim, 3, 1, 1),
            torch.nn.InstanceNorm2d(2 * base_dim, affine=True),
            torch.nn.GLU(dim=1),
        )
        self.block21 = torch.nn.Sequential(
            torch.nn.Conv2d(base_dim, 4 * base_dim, 3, 1, 1),
            torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
            torch.nn.GLU(dim=1),
        )
        self.block22 = torch.nn.Sequential(
            torch.nn.Conv2d(2 * base_dim, 4 * base_dim, 3, 1, 1),
            torch.nn.InstanceNorm2d(4 * base_dim, affine=True),
            torch.nn.GLU(dim=1),
        )
        self.block31 = torch.nn.Sequential(
            torch.nn.Conv2d(2 * base_dim, 8 * base_dim, 3, 1, 1),
            torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
            torch.nn.GLU(dim=1),
        )
        self.block32 = torch.nn.Sequential(
            torch.nn.Conv2d(4 * base_dim, 8 * base_dim, 3, 1, 1),
            torch.nn.InstanceNorm2d(8 * base_dim, affine=True),
            torch.nn.GLU(dim=1),
        )
        self.final_conv = torch.nn.Conv2d(4 * base_dim, out_dim, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor
    ) -> torch.Tensor:
        y = self.block11(x * mask)
        y = self.block12(y * mask)
        y += self.mlp1(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block21(y * mask)
        y = self.block22(y * mask)
        y += self.mlp2(time_emb).unsqueeze(-1).unsqueeze(-1)
        y = self.block31(y * mask)
        y = self.block32(y * mask)
        y = self.final_conv(y * mask)
        return (y * mask).sum((2, 3)) / (mask.sum((2, 3)) * x.shape[2])
