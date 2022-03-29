from spindletorch.unet.network import UNet, ResUNet, UNet3Plus


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
