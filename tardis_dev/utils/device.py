from typing import Optional

import torch


def get_device(device: Optional[str] = 0):
    """
    Return device that can be used for training or predictions

    Args:
        device (str, int): Device name or ID.

    Returns:
        str, torch.device: Device type.
    """
    if device == "gpu":  # Load GPU ID 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device('cuda:0')
    elif device == 'cpu':  # Load CPU
        device = torch.device('cpu')
    elif device_is_str(device):  # Load specific GPU ID
        device = torch.device(f'cuda:{int(device)}')
    elif device == 'mps':  # Load Apple silicon
        device = torch.device('mps')
    return device


def device_is_str(device: Optional[str] = 0) -> bool:
    """
    Check if used device is convertable to int value

    Args:
        device: (str, int): Device ID.

    Returns:
        bool: Check for input string/int
    """
    try:
        int(device)
        if isinstance(int(device), int):
            return True
        else:
            return False
    except ValueError:
        return False
