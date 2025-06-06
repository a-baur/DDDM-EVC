# Adapted from https://github.com/openai/jukebox

import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F

import modules.vqvae.dist as dist
from config import F0VQConfig


class BottleneckBlock(nn.Module):
    def __init__(self, k_bins: int, emb_width: int, mu: float) -> None:
        super().__init__()
        self.k_bins = k_bins
        self.emb_width = emb_width
        self.mu = mu

        self.init: bool = False
        self.register_k_buffer()

        self.threshold = 1.0

    def register_k_buffer(self) -> None:
        """
        Register k buffer. k represents the discrete latent space.
        :return:
        """
        self.register_buffer("k", torch.zeros(self.k_bins, self.emb_width))

    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tile x to have at least k_bins rows.

        :param x: Input tensor of shape (dim, embedding_width)
        :return: Tiled tensor of shape (k_bins, embedding_width)
        """
        d, ew = x.shape
        if d < self.k_bins:
            n_repeats = (self.k_bins + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_k(self, x: torch.Tensor) -> None:
        """
        Initialize k using random vectors from x.

        :param x: Input tensor of shape (dim, embedding_width)
        :return: None
        """
        self.init = True

        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][: self.k_bins]
        dist.broadcast(_k_rand, 0)

        self.k = _k_rand
        self.k_sum = self.k
        self.k_elem = torch.ones(self.k_bins, device=self.k.device)

        assert self.k.shape == (self.k_bins, self.emb_width)

    def restore_k(self, num_tokens: int = None, threshold: float = 1.0) -> None:
        """
        Restore k using random vectors from x.
        The number of tokens is used to calculate
        the expected usage of each bin.
        The threshold is used to determine if a bin is used.

        :param num_tokens: Number of tokens
        :param threshold: Threshold
        :return:
        """
        self.init = True
        assert self.k.shape == (self.k_bins, self.emb_width)

        self.k_sum = self.k.clone()
        self.k_elem = torch.ones(self.k_bins, device=self.k.device)

        if num_tokens is not None:
            expected_usage = num_tokens / self.k_bins
            self.k_elem.data.mul_(expected_usage)
            self.k_sum.data.mul_(expected_usage)

        self.threshold = threshold

    @torch.no_grad()  # type: ignore
    def update_k(self, x: torch.Tensor, x_l: torch.Tensor) -> dict:
        """

        :param x: Input tensor of shape (dim, embedding_width)
        :param x_l: Latent code tensor of shape (dim)
        :return: Dictionary of metrics
        """
        # Calculate new centres (k_bins, N * L)
        # - One hot encode x_l
        x_l_onehot = torch.zeros(self.k_bins, x.shape[0], device=x.device)
        x_l_onehot.scatter_(0, x_l.view(1, x.shape[0]), 1)

        # - Calculate sum and number of latent codes per bin
        _k_sum = torch.matmul(x_l_onehot, x)  # k_bins, w
        _k_elem = x_l_onehot.sum(dim=-1)  # k_bins

        # - Randomly sample from x to replace unused codebook bins
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][: self.k_bins]

        # Broadcast and reduce for distributed training
        dist.broadcast(_k_rand, 0)
        dist.all_reduce(_k_sum)
        dist.all_reduce(_k_elem)

        # Update centres per minibatch using exponential moving average
        # - Update k_sum and k_elem
        old_k = self.k
        self.k_sum = self.mu * self.k_sum + (1.0 - self.mu) * _k_sum  # w, k_bins
        self.k_elem = self.mu * self.k_elem + (1.0 - self.mu) * _k_elem  # k_bins

        # - Update k. If bin is sufficiently used, use new centre, else random centre
        usage = (self.k_elem.view(self.k_bins, 1) >= self.threshold).float()
        _k_new = self.k_sum.view(self.k_bins, self.emb_width) / self.k_elem.view(
            self.k_bins, 1
        )
        self.k = usage * _k_new + (1 - usage) * _k_rand

        # - Calculate metrics
        _k_prob = _k_elem / torch.sum(_k_elem)
        entropy = -torch.sum(_k_prob * torch.log(_k_prob + 1e-8))
        used_curr = (_k_elem >= self.threshold).sum()
        usage = torch.sum(usage)
        dk = torch.norm(self.k - old_k) / np.sqrt(np.prod(old_k.shape))
        return dict(entropy=entropy, used_curr=used_curr, usage=usage, dk=dk)

    def preprocess(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess x by reshaping and normalising.

        :param x: Input tensor of shape (dim, embedding_width)
        :return: Tuple of preprocessed tensor and prenorm
        """
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  # x_en = (N * L, w), k_j = (w, k_bins)

        if x.shape[-1] == self.emb_width:
            prenorm = torch.norm(x - torch.mean(x)) / np.sqrt(np.prod(x.shape))
        elif x.shape[-1] == 2 * self.emb_width:
            x1, x2 = x[..., : self.emb_width], x[..., self.emb_width :]
            prenorm = (torch.norm(x1 - torch.mean(x1)) / np.sqrt(np.prod(x1.shape))) + (
                torch.norm(x2 - torch.mean(x2)) / np.sqrt(np.prod(x2.shape))
            )

            # Normalise
            x = x1 + x2
        else:
            raise ValueError(
                f"Expected {x.shape[-1]} to be (1 or 2) * {self.emb_width}"
            )
        return x, prenorm

    @staticmethod
    def postprocess(
        x_l: torch.Tensor, x_d: torch.Tensor, x_shape: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Postprocess latent code and decoded tensor.

        :param x_l: Latent code tensor of shape (dim)
        :param x_d: Decoded tensor of shape (dim, embedding_width)
        :param x_shape: Shape of x
        :return: Tuple of postprocessed latent code and decoded tensor
        """
        # [NT, C] -> NTC -> NCT
        N, T = x_shape
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()
        x_l = x_l.view(N, T)
        return x_l, x_d

    def quantise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantise x by computing the latent code x_l.
        x_l is the discreet embedding of the closest bin k.
        Fit is the mean squared distance between x and the closest embedding.

        :param x: Input tensor of shape (dim, embedding_width)
        :return: Tuple of latent code and fit
        """
        # Calculate latent code x_l
        k_w = self.k.t()
        distance = (
            torch.sum(x**2, dim=-1, keepdim=True)
            - 2 * torch.matmul(x, k_w)
            + torch.sum(k_w**2, dim=0, keepdim=True)
        )  # (N * L, b)
        min_distance, x_l = torch.min(distance, dim=-1)
        fit = torch.mean(min_distance)
        return x_l, fit

    def dequantise(self, x_l: torch.Tensor) -> torch.Tensor:
        """
        Dequantise x_l by computing the decoded tensor x_d
        as the closest embedding in k.

        :param x_l: Latent code tensor of shape (dim)
        :return: Decoded tensor of shape (dim, embedding_width)
        """
        x = F.embedding(x_l, self.k)
        return x

    def forward(
        self, x: torch.Tensor, update_k: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Forward pass through bottleneck"""
        # Encode x to latent code
        N, width, T = x.shape
        x, prenorm = self.preprocess(x)
        if update_k and not self.init:
            self.init_k(x)
        x_l, fit = self.quantise(x)

        # Decode latent code to x
        x_d = self.dequantise(x_l)
        if update_k and self.training:
            update_metrics = self.update_k(x, x_l)
        else:
            update_metrics = {}

        # Calculate commitment loss
        commit_loss = torch.norm(x_d.detach() - x) ** 2 / np.prod(x.shape)

        # Postprocess x_l and x_d
        x_d = x + (x_d - x).detach()  # Passthrough
        x_l, x_d = self.postprocess(x_l, x_d, (N, T))

        return x_l, x_d, commit_loss, dict(fit=fit, pn=prenorm, **update_metrics)


class Bottleneck(nn.Module):
    """
    Pass series of inputs through bottleneck blocks.
    """

    def __init__(self, cfg: F0VQConfig) -> None:
        super().__init__()
        self.levels = cfg.levels
        self.level_blocks = nn.ModuleList()
        for _ in range(self.levels):
            self.level_blocks.append(BottleneckBlock(cfg.k_bins, cfg.emb_dim, cfg.mu))

    def encode(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Encode input tensors.

        :param xs: List of input tensors
        :return: List of latent codes
        """
        zs = [level_block.encode(x) for (level_block, x) in zip(self.level_blocks, xs)]
        return zs

    def decode(
        self, zs: list[torch.Tensor], start_level: int = 0, end_level: int = None
    ) -> list[torch.Tensor]:
        """
        Decode latent codes.

        Use start_level and end_level to decode a subset of the latent codes.

        :param zs: List of latent codes
        :param start_level: Start of the subset of latent codes to decode
        :param end_level: End of the subset of latent codes to decode
        :return: List of decoded tensors
        """
        if end_level is None:
            end_level = self.levels
        xs_quantised = [
            level_block.decode(z)
            for (level_block, z) in zip(self.level_blocks[start_level:end_level], zs)
        ]
        return xs_quantised

    def forward(
        self, xs: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[dict]]:
        """
        Pass input tensors through bottleneck blocks.

        :param xs: List of input tensors
        :return: Tuple of latent codes, decoded tensors, commitment losses, and metrics
        """
        zs, xs_quantised, commit_losses, metrics = [], [], [], []
        for level in range(self.levels):
            level_block = self.level_blocks[level]
            x = xs[level]
            z, x_quantised, commit_loss, metric = level_block(x, update_k=self.training)
            zs.append(z)
            if not self.training:
                # Be extra paranoid and make sure the encoder weights can't
                # change from straight-through estimator
                x_quantised = x_quantised.detach()
            xs_quantised.append(x_quantised)
            commit_losses.append(commit_loss)
            if self.training:
                metrics.append(metric)
        return zs, xs_quantised, commit_losses, metrics
