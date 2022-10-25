from torch.nn import init


def weights_init_kaiming(m):
    """
    Kaiming weight and bias initialization.

    Args:
        m: CNN block.
    """
    classname = m.__class__.__name__

    if classname.find('Conv3d') != -1:
        init.kaiming_normal_(tensor=m.weight.data,
                             a=0,
                             mode='fan_in')
    elif classname.find('Conv2d') != -1:
        init.kaiming_normal_(tensor=m.weight.data,
                             a=0,
                             mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(tensor=m.weight.data,
                     mean=1.0,
                     std=0.02)
        init.constant_(tensor=m.bias.data,
                       val=0.0)
    elif classname.find('GroupNorm') != -1:
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
