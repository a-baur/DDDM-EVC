import numpy as np
import torch


class BaseModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def nparams(self) -> int:
        """Number of trainable parameters in the model."""
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params

    def relocate_input(self, x: list) -> list:
        """Relocate input tensors to the same device as the model."""
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x
