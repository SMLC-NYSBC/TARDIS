from tardis_dev.spindletorch.utils.build_cnn import ResUNet, UNet, UNet3Plus, FNet


def build_network(network_type: str,
                  classification: bool,
                  in_channel=1,
                  out_channel=1,
                  img_size=64,
                  dropout=None,
                  num_conv_layers=5,
                  conv_scaler=64,
                  conv_kernel=3,
                  conv_padding=1,
                  maxpool_kernel=2,
                  layer_components='3gcl',
                  num_group=8,
                  prediction=False):
    """
    Wrapper for building CNN model

    Wrapper take CNN parameter and predefined network type (e.g. unet, etc.), 
    and build CNN model.

    Args:
        network_type (str): Name of network [unet, resunet, unet3plus, big_unet, fnet]
        classification (bool): If True Unet3Plus classified network output before loss
        in_channel (int): Number of input channels
        out_channel (int): Number of output channels
        img_size (int): Image size used for training inference
        dropout (float, optional): If float dropout is used
        num_conv_layers (int): Number of convolution layers
        conv_scaler (int): Convolution multiplayer used in each layer
        layer_components (str): (2 or 3),b,g,c,l,r type of operation and order for
            each convolution
        num_group (int): Number of group used for groupnorm
        prediction (bool): If True, network output softmax of the prediction
    """

    if network_type == 'unet':
        return UNet(in_channels=in_channel,
                    out_channels=out_channel,
                    img_patch_size=img_size,
                    dropout=dropout,
                    conv_kernel=conv_kernel,
                    padding=conv_padding,
                    pool_kernel=maxpool_kernel,
                    num_group=num_group,
                    num_conv_layer=num_conv_layers,
                    conv_layer_scaler=conv_scaler,
                    layer_components=layer_components,
                    prediction=prediction)
    elif network_type == 'resunet':
        return ResUNet(in_channels=in_channel,
                       out_channels=out_channel,
                       img_patch_size=img_size,
                       dropout=dropout,
                       num_conv_layer=num_conv_layers,
                       conv_layer_scaler=conv_scaler,
                       conv_kernel=conv_kernel,
                       padding=conv_padding,
                       pool_kernel=maxpool_kernel,
                       num_group=num_group,
                       layer_components=layer_components,
                       prediction=prediction)
    elif network_type == 'unet3plus':
        return UNet3Plus(in_channels=in_channel,
                         out_channels=out_channel,
                         classifies=classification,
                         img_patch_size=img_size,
                         conv_kernel=conv_kernel,
                         padding=conv_padding,
                         pool_kernel=maxpool_kernel,
                         num_conv_layer=num_conv_layers,
                         conv_layer_scaler=conv_scaler,
                         layer_components=layer_components,
                         num_group=num_group,
                         prediction=prediction)
    elif network_type == 'fnet':
        return FNet(in_channels=in_channel,
                    out_channels=out_channel,
                    img_patch_size=img_size,
                    conv_kernel=conv_kernel,
                    padding=conv_padding,
                    pool_kernel=maxpool_kernel,
                    num_conv_layer=num_conv_layers,
                    conv_layer_scaler=conv_scaler,
                    layer_components=layer_components,
                    num_group=num_group,
                    prediction=prediction)
    else:
        return None
