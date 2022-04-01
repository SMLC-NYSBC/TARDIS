from typing import Optional
import torch


def get_device(device: Optional[str] = 0):
    """
    RETURN DEVICE THAT CAN BE USED FOR TRAINING/PREDICTIONS

    Args:
        device: Device name or ID
    """
    if device == "gpu":  # Load GPU ID 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cuda:0'
    elif isinstance(device, int):  # Load specific GPU ID
        device = f'cuda:{device}'
    elif device == 'cpu':  # Load CPU
        device = 'cpu'
    else:
        device = 'cpu'

    return device
