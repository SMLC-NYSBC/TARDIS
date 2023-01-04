from tardis.spindletorch.utils.build_cnn import FNet, ResUNet, UNet, UNet3Plus
from tardis.utils.errors import TardisError


def build_cnn_network(network_type: str,
                      structure: dict,
                      img_size: int,
                      prediction: bool):
    """
    Wrapper for building CNN model

    Wrapper take CNN parameter and predefined network type (e.g. unet, etc.),
    and build CNN model.

    Args:
        network_type (str): Name of network [unet, resunet, unet3plus, big_unet, fnet].
        structure (dict): Dictionary with all setting to build CNN.
        img_size (int): Image patch size used for CNN.
        prediction (bool): If true, build CNN in prediction patch.
    """
    assert network_type in ['unet', 'resunet', 'unet3plus', 'fnet'], \
        TardisError('141',
                    'tardis/spindletorch/spindletorch.py',
                    f'Wrong CNN network name {network_type}')

    if network_type == 'unet':
        return UNet(in_channels=structure['in_channel'],
                    out_channels=structure['out_channel'],
                    img_patch_size=img_size,
                    dropout=structure['dropout'],
                    conv_kernel=structure['conv_kernel'],
                    padding=structure['conv_padding'],
                    pool_kernel=structure['maxpool_kernel'],
                    num_group=structure['num_group'],
                    num_conv_layer=structure['num_conv_layers'],
                    conv_layer_scaler=structure['conv_scaler'],
                    layer_components=structure['layer_components'],
                    prediction=prediction)
    elif network_type == 'resunet':
        return ResUNet(in_channels=structure['in_channel'],
                       out_channels=structure['out_channel'],
                       img_patch_size=img_size,
                       dropout=structure['dropout'],
                       num_conv_layer=structure['num_conv_layers'],
                       conv_layer_scaler=structure['conv_scaler'],
                       conv_kernel=structure['conv_kernel'],
                       padding=structure['conv_padding'],
                       pool_kernel=structure['maxpool_kernel'],
                       num_group=structure['num_group'],
                       layer_components=structure['layer_components'],
                       prediction=prediction)
    elif network_type == 'unet3plus':
        return UNet3Plus(in_channels=structure['in_channel'],
                         out_channels=structure['out_channel'],
                         classifies=structure['classification'],
                         dropout=structure['dropout'],
                         img_patch_size=img_size,
                         conv_kernel=structure['conv_kernel'],
                         padding=structure['conv_padding'],
                         pool_kernel=structure['maxpool_kernel'],
                         num_conv_layer=structure['num_conv_layers'],
                         conv_layer_scaler=structure['conv_scaler'],
                         layer_components=structure['layer_components'],
                         num_group=structure['num_group'],
                         prediction=prediction)
    elif network_type == 'fnet':
        return FNet(in_channels=structure['in_channel'],
                    out_channels=structure['out_channel'],
                    img_patch_size=img_size,
                    conv_kernel=structure['conv_kernel'],
                    padding=structure['conv_padding'],
                    pool_kernel=structure['maxpool_kernel'],
                    num_conv_layer=structure['num_conv_layers'],
                    conv_layer_scaler=structure['conv_scaler'],
                    layer_components=structure['layer_components'],
                    num_group=structure['num_group'],
                    prediction=prediction)
    else:
        return None
