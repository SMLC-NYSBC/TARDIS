from tardis.spindletorch.unet.network import ResUNet, UNet, UNet3Plus, Big_UNet, FNet


def build_network(network_type: str,
                  classification: bool,
                  in_channel=1,
                  out_channel=1,
                  img_size=64,
                  dropout=None,
                  no_conv_layers=5,
                  conv_multiplayer=64,
                  conv_kernel=3,
                  conv_padding=1,
                  maxpool_kernel=2,
                  layer_components='3gcl',
                  no_groups=8,
                  prediction=False):
    """
    MAIN MODULE FOR BUILDING CNN

    Args:
        network_type: Name of network [unet, resunet, unet3plus, big_unet, fnet]
        classification: If True Unet3Plus classified network output before loss
        in_channel: Number of input channels
        out_channel: Number of output channels
        img_size: Image size used for training inference
        dropout: If float dropout is used
        no_conv_layers: Number of convolution layers
        conv_multiplayer: Convolution multiplayer used in each layer
        layer_components: (2 or 3),b,g,c,l,r type of operation and order for
            each convolution
        no_groups: Number of group used for groupnorm
        prediction: If True, network output softmax of the prediction
    """

    if network_type == 'unet':
        return UNet(in_channels=in_channel,
                    out_channels=out_channel,
                    patch_size=img_size,
                    dropout=dropout,
                    conv_kernel=conv_kernel,
                    padding=conv_padding,
                    pool_kernel=maxpool_kernel,
                    no_groups=no_groups,
                    no_conv_layer=no_conv_layers,
                    conv_layer_multiplayer=conv_multiplayer,
                    layer_components=layer_components,
                    prediction=prediction)
    elif network_type == 'resunet':
        return ResUNet(in_channels=in_channel,
                       out_channels=out_channel,
                       patch_size=img_size,
                       dropout=dropout,
                       no_conv_layer=no_conv_layers,
                       conv_layer_multiplayer=conv_multiplayer,
                       conv_kernel=conv_kernel,
                       padding=conv_padding,
                       pool_kernel=maxpool_kernel,
                       layer_components=layer_components,
                       prediction=prediction)
    elif network_type == 'unet3plus':
        return UNet3Plus(in_channels=in_channel,
                         out_channels=out_channel,
                         classifies=classification,
                         patch_size=img_size,
                         conv_kernel=conv_kernel,
                         padding=conv_padding,
                         pool_kernel=maxpool_kernel,
                         no_conv_layer=no_conv_layers,
                         conv_layer_multiplayer=conv_multiplayer,
                         layer_components=layer_components,
                         no_groups=no_groups,
                         prediction=prediction)
    elif network_type == 'big_unet':
        return Big_UNet(in_channels=in_channel,
                        out_channels=out_channel,
                        patch_size=img_size,
                        conv_kernel=conv_kernel,
                        padding=conv_padding,
                        pool_kernel=maxpool_kernel,
                        no_conv_layer=no_conv_layers,
                        conv_layer_multiplayer=conv_multiplayer,
                        layer_components=layer_components,
                        no_groups=no_groups,
                        prediction=prediction)
    elif network_type == 'fnet':
        return FNet(in_channels=in_channel,
                    out_channels=out_channel,
                    patch_size=img_size,
                    conv_kernel=conv_kernel,
                    padding=conv_padding,
                    pool_kernel=maxpool_kernel,
                    no_conv_layer=no_conv_layers,
                    conv_layer_multiplayer=conv_multiplayer,
                    layer_components=layer_components,
                    no_groups=no_groups,
                    prediction=prediction)
    else:
        return None
