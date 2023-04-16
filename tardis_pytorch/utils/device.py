#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional

import torch


def get_device(device: Optional[str] = 0) -> torch.device:
    """
    Return device that can be used for training or predictions

    Args:
        device (str, int): Device name or ID.

    Returns:
        torch.device: Device type.
    """
    if device == "gpu":  # Load GPU ID 0
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif device == "cpu":  # Load CPU
        device = torch.device("cpu")
    elif device_is_str(device):  # Load specific GPU ID
        device = torch.device(f"cuda:{int(device)}")
    elif device == "mps":  # Load Apple silicon
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def device_is_str(device: Optional[str] = 0) -> bool:
    """
    Check if used device is convertible to int value

    Args:
        device (str, int): Device ID.

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
