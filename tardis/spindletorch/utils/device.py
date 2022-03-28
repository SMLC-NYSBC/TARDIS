import torch


def get_device(device=None):
    """
    Return device that can be used for training/predictions

    Args:
        device: If indicated then overnight automatic selection of the device

    author: Robert Kiewisz
    """
    if device is None or device == "gpu":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda':
            for i in range(torch.cuda.device_count()):
                device = torch.device('cuda:{}'.format(i))
    else:
        device = device
    return device
