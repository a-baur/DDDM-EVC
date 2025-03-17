import torch
import torch.nn.functional as F
from torch import nn

import util
from models import DDDMInput


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x.mT, (self.channels,), self.gamma, self.beta, self.eps)
        return x.mT


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Linear(filter_channels, 1)

        if gin_channels != 0:
            self.cond = nn.Linear(gin_channels, in_channels)

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: torch.Tensor = None
    ) -> torch.Tensor:
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g.mT).mT
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj((x * x_mask).mT).mT
        return x * x_mask


class DurationControl(nn.Module):
    """
    Duration control module for DDDM model.

    Based on https://github.com/hs-oh-prml/DurFlexEVC/blob/main/models/evc/durflex/utils.py#L39
    """

    def __init__(
        self,
        in_dim: int,
        gin_channels: int,
    ) -> None:
        super().__init__()
        self.duration_predictor = DurationPredictor(
            in_dim, 256, 3, 0.5, gin_channels=gin_channels
        )

    def forward(
        self,
        x: DDDMInput,
        g: torch.Tensor,
        return_loss: bool = False,
    ) -> DDDMInput | tuple[DDDMInput, torch.Tensor]:
        assert x.phonemes is not None

        # unit pooling
        units, dur = self._deduplicate(x.phonemes * x.mask.squeeze(1))
        unit_mapping = self._get_unit_mapping(dur).unsqueeze(1)

        u_mask = units != 0
        u_emb_c = self._unit_pooling(x.emb_content, unit_mapping)
        u_emb_p = self._unit_pooling(x.emb_pitch, unit_mapping)

        # frame-level expansion
        dur_pred = self.duration_predictor(
            u_emb_c.detach(),
            u_mask.unsqueeze(1),
            g,
        ).squeeze(1)
        # dur_pred = torch.exp(log_dur_pred) * u_mask
        dur_pred = torch.ceil(dur_pred * u_mask)
        unit_mapping = self._get_unit_mapping(dur_pred).unsqueeze(1)

        x.mask = (unit_mapping != 0).detach()
        x.emb_content = self._frame_expansion(u_emb_c, unit_mapping)
        x.emb_pitch = self._frame_expansion(u_emb_p, unit_mapping)

        new_len = unit_mapping.size(2)
        if return_loss:
            loss = F.l1_loss(dur_pred, dur)
            x.mel = F.interpolate(x.mel, size=new_len, mode="linear")
            return x, loss
        else:
            x.mel = F.interpolate(x.mel, size=new_len, mode="linear")
            return x

    @staticmethod
    def _unit_pooling(x: torch.Tensor, unit_mapping: torch.Tensor) -> torch.Tensor:
        """Pool phoneme embeddings to unit-level.

        :example:
        >>> x = torch.tensor([[[1, 2, 3],
        ...                    [1, 2, 3],
        ...                    [1, 2, 3]]])
        >>> unit_mapping = torch.tensor([[1, 1, 2]]).unsqueeze(1)
        >>> DurationControl._unit_pooling(x, unit_mapping)
        tensor([[[1.5000, 3.0000],
                 [1.5000, 3.0000],
                 [1.5000, 3.0000]]])

        :param x: Embedding tensor of size (B, C, T)
        :param d: Duration tensor of size (B, T')
        :return: Pooled embedding tensor of size (B, C, T')
        """
        max_len = unit_mapping.max() + 1
        B, C, T = x.shape

        # create unit-level zeroes and use unit mapping
        # to add up embeddings for each unit
        units = x.new_zeros((B, C, max_len))
        units = units.scatter_add_(2, unit_mapping.expand(-1, C, -1), x)

        # compute the number of frames that each unit spans
        all_ones = x.new_ones((B, 1, T))
        unit_counts = x.new_zeros((B, 1, max_len))
        unit_counts = unit_counts.scatter_add_(2, unit_mapping, all_ones)

        # remove padding frame
        units = units[:, :, 1:]
        unit_counts = unit_counts[:, :, 1:]

        # compute the mean of embeddings for each unit
        return units / torch.clamp(unit_counts, min=1)

    @staticmethod
    def _frame_expansion(x: torch.Tensor, unit_mapping: torch.Tensor) -> torch.Tensor:
        """Expand phoneme embeddings to frame-level.

        The duration tensor expands the i-th unit
        to d[i] frames:

        :example:
        >>> x = torch.tensor([[[1, 2],
        ...                    [3, 4],
        ...                    [5, 6]]])
        >>> um = torch.tensor([[1, 1, 2]]).unsqueeze(1)
        >>> DurationControl._frame_expansion(x, um)
        tensor([[[1, 1, 2],
                 [3, 3, 4],
                 [5, 5, 6]]])

        If the duration tensor sums to different lengths
        across the batch, the output tensor will be padded
        with zeros using padding frame at index 0:

        :example:
        >>> x = torch.tensor([[[1, 2],
        ...                    [3, 4],
        ...                    [5, 6]],
        ...                   [[7, 8],
        ...                    [9, 10],
        ...                    [11, 12]]])
        >>> um = torch.tensor([[1, 1, 2, 2, 2, 2],
        ...                    [1, 2, 2, 0, 0, 0]]).unsqueeze(1)
        >>> DurationControl._frame_expansion(x, um)
        tensor([[[ 1,  1,  2,  2,  2,  2],
                 [ 3,  3,  4,  4,  4,  4],
                 [ 5,  5,  6,  6,  6,  6]],
        <BLANKLINE>
                [[ 7,  8,  8,  0,  0,  0],
                 [ 9, 10, 10,  0,  0,  0],
                 [11, 12, 12,  0,  0,  0]]])

        :param x: Embedding tensor
        :param unit_mapping: Unit mapping tensor

        :return: Expanded embedding tensor
        """
        x = F.pad(x, [1, 0])  # zero-th frame is padding frame
        unit_mapping = unit_mapping.repeat((1, x.shape[1], 1))
        return torch.gather(x, 2, unit_mapping)

    def _deduplicate(self, ph: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deduplicate batched phoneme sequences."""
        units, durations = zip(
            *[self._deduplicate_sample(ph_sample) for ph_sample in ph]
        )

        max_len = max([u.size(0) for u in units])
        units = util.pad_tensors_to_length(units, max_len)
        durations = util.pad_tensors_to_length(durations, max_len)

        return torch.stack(units), torch.stack(durations)

    @staticmethod
    def _deduplicate_sample(
        ph_sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deduplicate phoneme sequence.

        :example:
        >>> ph = torch.tensor([1, 1, 2, 3, 3, 3, 4, 4, 4, 4])
        >>> DurationControl._deduplicate_sample(ph_sample)
        (tensor([1, 2, 3, 4]), tensor([2, 1, 3, 4]))

        :param ph_sample: Phoneme sequence tensor
        :return: Deduplicated phoneme sequence and duration tensor
        """
        # boolean mask for value changes
        diff = torch.cat(
            [
                torch.tensor([True], device=ph_sample.device),
                ph_sample[1:] != ph_sample[:-1],
            ]
        )

        # unique units are the values where diff is True
        units = ph_sample[diff]

        # get the indices where changes occur
        idx = diff.nonzero(as_tuple=False).squeeze(1)

        # add the last index to the end
        idx = torch.cat(
            [idx, torch.tensor([ph_sample.size(0)], device=ph_sample.device)]
        )

        # get the counts of each unit
        counts = idx[1:] - idx[:-1]

        return units, counts

    @staticmethod
    def _get_unit_mapping(d: torch.Tensor) -> torch.Tensor:
        """Get unit mapping from durations.

        The output tensor maps the phoneme units to
        the corresponding frame indices.

        Indexing starts from 1 to accommodate
        padding frame at index 0.

        :example:
        >>> dur = torch.tensor([[3, 2, 1], [2, 1, 2]])
        >>> DurationControl._get_unit_mapping(dur)
        tensor([[1, 1, 1, 2, 2, 3],
                [1, 1, 2, 3, 3, 0]])

        :param d: Duration tensor of size (B, T')
        :return: Unit mapping tensor of size (B, T)
        """
        # Examples with duration tensor d = [2, 2, 3]:
        # Create unit mapping tensor
        # [1]
        # [2]
        # [3]
        unit_idx = torch.arange(1, d.shape[-1] + 1, device=d.device)[None, :, None]

        # Create token mask
        #
        # Cumulative sum of durations to get the index of unit changes:
        # dur_cumsum =      [      2,    4,      7]
        # dur_cumsum_prev = [0,    2,    4]
        # pos_idx =         [0, 1, 2, 3, 4, 5, 6]
        #
        #      pos_idx >= dur_cumsum_prev & pos_idx < dur_cumsum
        #
        #                   [1, 1, 0, 0, 0, 0, 0]
        # token_mask =      [0, 0, 1, 1, 0, 0, 0]
        #                   [0, 0, 0, 0, 1, 1, 1]
        dur_cumsum = torch.cumsum(d, 1)
        dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)
        pos_idx = torch.arange(d.sum(-1).max().item(), device=d.device)[None, None, :]
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (
            pos_idx < dur_cumsum[:, :, None]
        )

        # Compute unit mapping
        # [1]   [1,1,0,0,0,0,0]   [1,1,0,0,0,0,0],
        # [2] * [0,0,1,1,0,0,0] = [0,0,2,2,0,0,0],
        # [3]   [0,0,0,0,1,1,1]   [0,0,0,0,3,3,3]
        #                              sum(1)
        #                         [1,1,2,2,3,3,3]
        return (unit_idx * token_mask.long()).sum(1)


def _test() -> None:
    import time

    # dc = DurationControl(10)
    # ph = torch.tensor(
    #     [
    #         [1, 1, 2, 3, 3, 3, 4, 4, 4, 4],
    #         [5, 5, 6, 6, 6, 7, 7, 8, 9, 9],
    #         [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
    #     ]
    # )
    # emb_c = torch.full((3, 1024, 10), 10)
    # emb_p = torch.full((3, 1024, 10), 10)
    # g = torch.randn(3, 512)
    #
    # durations, units = dc(ph, emb_c, emb_p, g)
    # print(durations)
    # print(units)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dc = DurationControl(1024, 512).to(device)
    x_mask = torch.full((32, 118), 1).unsqueeze(1).to(device).detach()
    ph = torch.full((32, 118), 1).to(device).detach()
    emb_c = torch.randn(32, 1024, 118).to(device)
    emb_p = torch.randn(32, 1024, 118).to(device)

    x = DDDMInput(
        audio=None,
        mel=None,
        mask=x_mask,
        emb_content=emb_c,
        emb_pitch=emb_p,
        phonemes=ph,
    )

    start = time.time()
    output = dc(x, None)
    print(time.time() - start)
    print(output.emb_content.shape)
    print(output.emb_pitch.shape)
    print(output.mask.shape)


if __name__ == "__main__":
    _test()
