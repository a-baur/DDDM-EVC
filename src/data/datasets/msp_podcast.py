import torch


class MSPPodcast(torch.utils.data.Dataset):
    def __init__(self, path: str) -> None:
        self.path = path

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.rand(1, 16000)

    def __len__(self) -> int:
        return 1000
