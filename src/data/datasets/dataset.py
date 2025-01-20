import torch
from torch.utils.data import Dataset

from config import Config


class AudioDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        cfg: Config,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.hop_length = cfg.data.mel_transform.hop_length
        self.segment_size = cfg.training.segment_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        sample_len = sample[0].size(1)

        # Pad waveforms to the same length
        padding_length = self.segment_size - sample_len
        padded_waveform = torch.nn.functional.pad(
            sample[0], (0, padding_length), mode="constant"
        ).data

        # Create length tensor
        mel_length = self.segment_size // self.hop_length
        if sample_len < mel_length:
            length = torch.LongTensor([sample_len // self.hop_length])
        else:
            length = torch.LongTensor([mel_length])

        return padded_waveform, length
