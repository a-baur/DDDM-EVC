import torch
import torch.nn as nn

import util
from config import WavenetDecoderConfig


class WaveNet(torch.nn.Module):
    """
    WaveNet model, as proposed in https://arxiv.org/abs/1609.03499.

    Implementation from:

    - https://github.com/hayeong0/DDDM-VC/blob/master/modules_sf/modules.py#L300
    - https://github.com/jaywalnut310/glow-tts/blob/master/modules.py#L68
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(
                gin_channels, 2 * hidden_channels * n_layers, 1
            )
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the WaveNet model.

        :param x: Input tensor
        :param x_mask: Mask tensor for padding
        :param g: Global conditioning tensor
        :return: Output tensor
        """
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = util.fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)


class Decoder(nn.Module):
    def __init__(self, cfg: WavenetDecoderConfig) -> None:
        super().__init__()
        self.in_channels = cfg.in_dim
        self.hidden_channels = cfg.hidden_dim
        self.kernel_size = cfg.kernel_size
        self.dilation_rate = cfg.dilation_rate
        self.n_layers = cfg.n_layers
        self.gin_channels = cfg.gin_channels

        self.pre = nn.Conv1d(cfg.in_dim, cfg.hidden_dim, 1)
        self.enc = WaveNet(
            cfg.hidden_dim,
            cfg.kernel_size,
            cfg.dilation_rate,
            cfg.n_layers,
            gin_channels=cfg.gin_channels,
        )
        self.proj = nn.Conv1d(cfg.hidden_dim, cfg.n_mel_channels, 1)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the WaveNet decoder.

        :param x: Input tensor
        :param x_mask: Mask tensor for padding
        :param g: Global conditioning tensor
        :return: Output tensor
        """
        x = self.pre(x * x_mask) * x_mask
        x = self.enc(x, x_mask, g=g)
        x = self.proj(x) * x_mask

        return x
