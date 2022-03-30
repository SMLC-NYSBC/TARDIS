from tardis.spindletorch.unet.network import UNet, ResUNet, UNet3Plus


def build_network(network_type: str,
                  classification: bool,
                  in_channel=1,
                  out_channel=1,
                  img_size=64,
                  dropout=None,
                  no_conv_layers=5,
                  conv_multiplayer=64,
                  layer_components='gcl',
                  no_groups=8,
                  prediction=False):
    """
    MAIN MODULE FOR BUILDING CNN

    Args:
        network_type: Name of network
        classification: If True Unet3Plus classified network output before loss
        in_channel: Number of input channels
        out_channel: Number of output channels
        img_size: Image size used for training inference
        dropout: If float dropout is used
        no_conv_layers: Number of convoltion layers
        conv_multiplayer: Convolution multiplayer used in each layer
        layer_components: b,g,c,l,r type of operation and order for each convolution
        no_groups: Number of group used for groupnorm
        prediction: If True, network output softmax of the prediction
    """
    if network_type == 'unet':
        model = UNet(in_channels=in_channel,
                     out_channels=out_channel,
                     patch_size=img_size,
                     dropout=dropout,
                     no_conv_layer=no_conv_layers,
                     conv_layer_multiplayer=conv_multiplayer,
                     layer_components=layer_components,
                     prediction=prediction)
    elif network_type == 'resunet':
        model = ResUNet(in_channels=in_channel,
                        out_channels=out_channel,
                        patch_size=img_size,
                        dropout=dropout,
                        no_conv_layer=no_conv_layers,
                        conv_layer_multiplayer=conv_multiplayer,
                        layer_components=layer_components,
                        prediction=prediction)
    elif network_type == 'unet3plus':
        model = UNet3Plus(in_channels=in_channel,
                          out_channels=out_channel,
                          classifies=classification,
                          patch_size=img_size,
                          no_conv_layer=no_conv_layers,
                          conv_layer_multiplayer=conv_multiplayer,
                          layer_components=layer_components,
                          no_groups=no_groups,
                          prediction=prediction)
    else:
        model = None

    return model
