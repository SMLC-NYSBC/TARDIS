from torch.nn import init


def weights_init_kaiming(m):
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


def init_weights(net, init_type='normal'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)
