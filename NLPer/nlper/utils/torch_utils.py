import torch

from typing import Union

AVAILABLE_GPU = torch.cuda.is_available()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def to_gpu(tensor: torch.Tensor) -> Union[torch.Tensor.cuda, torch.Tensor]:
    return tensor.cuda() if AVAILABLE_GPU else tensor
