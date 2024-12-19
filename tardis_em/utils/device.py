#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import torch


def get_device(device: str = "0") -> torch.device:
    """
    Determines and returns a PyTorch device based on the specified input.

    This function assesses the provided device string and determines which
    PyTorch device to use. Supported inputs include "gpu", "cpu", "mps", or
    specific GPU device indices. If the specified GPU or MPS devices are
    unavailable, it defaults to using the CPU device. The function
    ensures compatibility with the hardware and backend availability.

    :param device: A string representing the desired computational device.
        It can be "gpu", "cpu", "mps", or a string specifying a GPU index.
    :type device: str
    :return: A PyTorch device object representing the availability and
        compatibility of the requested device.
    :rtype: torch.device
    """
    if device == "gpu":  # Load GPU ID 0
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif device == "cpu":  # Load CPU
        device = torch.device("cpu")
    elif device_is_str(device):  # Load specific GPU ID
        if torch.cuda.is_available():
            if int(device) == -1:
                device = torch.device("cpu")
            else:
                device = torch.device(f"cuda:{int(device)}")
        else:
            device = torch.device("cpu")
    elif device == "mps":  # Load Apple silicon
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return device


def device_is_str(device: str = "0") -> bool:
    """
    Checks if the provided device identifier string represents a valid integer and determines
    if the input can be interpreted as a valid device representation.

    This function attempts to convert the provided string into an integer. If successful,
    the function confirms whether the converted value is an integer type and returns the
    result accordingly. If the string cannot be converted into an integer, it explicitly
    returns False.

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
