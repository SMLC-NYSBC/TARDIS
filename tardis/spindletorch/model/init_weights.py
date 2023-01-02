from torch.nn import init


def weights_init_kaiming(m):
    """
    Kaiming weight and bias initialization.

    Args:
        m: CNN block.
    """
    class_name = m.__class__.__name__

    if class_name.find('Conv3d') != -1:
        init.kaiming_normal_(tensor=m.weight.data,
                             a=0,
                             mode='fan_in')
    elif class_name.find('Conv2d') != -1:
        init.kaiming_normal_(tensor=m.weight.data,
                             a=0,
                             mode='fan_in')
    elif class_name.find('BatchNorm') != -1:
        init.normal_(tensor=m.weight.data,
                     mean=1.0,
                     std=0.02)
        init.constant_(tensor=m.bias.data,
                       val=0.0)
    elif class_name.find('GroupNorm') != -1:
        init.normal_(tensor=m.weight.data,
                     mean=1.0,
                     std=0.02)
        init.constant_(tensor=m.bias.data,
                       val=0.0)


def init_weights(net,
                 init_type='kaiming'):
    """
    Wrapper for network module initialization.

    Args:
        net: Network to initialized.
        init_type (str): type of initialization.

    Raises:
        NotImplementedError: _description_
    """
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(
            f'initialization method {init_type} is not implemented!')
