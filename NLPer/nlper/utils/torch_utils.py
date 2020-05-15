import torch


AVAILABLE_GPU = torch.cuda.is_available()


def get_device() -> object:
    """
    Returns the available device

    :return: device
    :rtype: object
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def to_gpu(tensor: torch.Tensor) -> torch.Tensor:
    """
    Transfers torch tensor to available device

    :param tensor: Tensor to be transferred to device
    :type tensor:  torch.Tensor
    :return:
    :rtype: torch.Tensor
    """
    return tensor.cuda() if AVAILABLE_GPU else tensor
