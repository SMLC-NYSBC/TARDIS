#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from typing import Union

import torch


def get_device(device: Union[str, list, tuple] = "0") -> torch.device:
    """
    Determines and returns a device object or list of device objects based on the input,
    with support for different configurations such as GPU, CPU, MPS (for Apple silicon),
    or specific device indices.

    This function is responsible for creating a `torch.device` object or a list of
    them, depending on the input device configuration. It supports various input
    types and values for device specification and ensures fallbacks when the
    desired devices are unavailable.

    :param device: Any of the following supported values:
        - "gpu": Uses the first GPU if available, otherwise falls back to CPU.
        - "cpu": Explicitly returns a device for the CPU.
        - "mps": For Apple silicon, utilizes the MPS device if available, defaults to CPU otherwise.
        - str: A specific GPU ID as a string (e.g., "0"). Falls back to CPU if the GPU is unavailable.
        - int: A specific GPU ID in integer form.
        - list or tuple: A collection of GPU IDs for handling multiple devices. The function returns a list of `torch.device` objects corresponding to the indices.

    :return: A `torch.device` object or a list of `torch.device` objects depending on the input configuration.
    """
    device_ = None

    if device == "gpu":  # Load GPU ID 0
        if torch.cuda.is_available():
            device_ = torch.device("cuda:0")
        else:
            device_ = torch.device("cpu")
    elif device == "cpu":  # Load CPU
        device_ = torch.device("cpu")
    elif device_is_str(device):  # Load specific GPU ID
        if torch.cuda.is_available():
            if int(device) == -1:
                device_ = torch.device("cpu")
            else:
                device_ = torch.device(f"cuda:{int(device)}")
        else:
            device_ = torch.device("cpu")
    elif isinstance(device, list) or isinstance(device, tuple):
        if torch.cuda.is_available():
            device_ = []
            for i in device:
                if device_is_str(i):
                    device_.append(int(i))
    elif device == "mps":  # Load Apple silicon
        if torch.backends.mps.is_available():
            device_ = torch.device("mps")
        else:
            device_ = torch.device("cpu")
    return device_


def device_is_str(device: str = "0") -> bool:
    """
    Checks if the provided device identifier string represents a valid integer and determines
    if the input can be interpreted as a valid device representation.

    :param device: The device identifier as a string. Defaults to "0".
    :type device: str

    :return: True if the input string can be cast to an integer and is a valid integer type;
        otherwise, False.
    :rtype: bool
    """
    try:
        int(device)
        if isinstance(int(device), int):
            return True
        else:
            return False
    except ValueError:
        return False
