import torch


def get_device(device=None):
    """
    RETURN DEVICE THAT CAN BE USED FOR TRAINING/PREDICTIONS

    Args:
        device: If indicated then overnight automatic selection of the device
    """
    if device is None or device == "gpu":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if device == 'cuda':
            for i in range(torch.cuda.device_count()):
                device = torch.device('cuda:{}'.format(i))
    else:
        device = device
    return device
