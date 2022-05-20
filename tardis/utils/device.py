from typing import Optional
import torch


def get_device(device: Optional[str] = 0):
    """
    RETURN DEVICE THAT CAN BE USED FOR TRAINING/PREDICTIONS

    Args:
        device: Device name or ID
    """
    if device == "gpu":  # Load GPU ID 0
        torch.set_num_threads(4)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cuda:0'
    elif device == 'cpu':  # Load CPU
        device = 'cpu'
    elif device_is_str(device):  # Load specific GPU ID
        torch.set_num_threads(4)
        device = f'cuda:{int(device)}'
    elif device == 'mps':  # Load Apple silicon
        device = 'mps'
    return device


def device_is_str(device):
    """
    CHECK IF USED DEVICE IS CONVERTABLE TO INT VALUE

    Args:
        device: Device ID
    """
    try:
        int(device)
        if isinstance(int(device), int):
            return True
        else:
            return False
    except ValueError:
        return False
