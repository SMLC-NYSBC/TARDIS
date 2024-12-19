#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from tardis_em.cnn.utils.build_cnn import FNet, ResUNet, UNet, UNet3Plus
from tardis_em.utils.errors import TardisError


def build_cnn_network(
    network_type: str, structure: dict, img_size: int, prediction: bool
):
    """
    Builds and returns an instance of a CNN-based network based on the provided
    network type and configuration structure. The function supports various types
    of networks such as UNet, ResUNet, UNet3Plus, FNet, and FNet with attention
    mechanism. Each network type is associated with specific architectural
    parameters outlined in the structure dictionary. The function validates the
    input network type and configures the network based on the given attributes.

    :param network_type: The type of CNN network to build. Possible values are
        "unet", "resunet", "unet3plus", "fnet", or "fnet_attn".
    :param structure: A dictionary containing the structural parameters for the
        network, including attributes like in_channel, out_channel, dropout,
        conv_kernel, etc.
    :param img_size: An integer specifying the image patch size used in the
        network. It defines the dimensions of the input image patches.
    :param prediction: A boolean indicating whether the network is used in
        prediction mode or not. Affects specific layers or configurations in
        the network instance.

    :return: An instance of the specified CNN network configured with the
        provided attributes. Returns None if the input network type is invalid.
    """
    if network_type not in ["unet", "resunet", "unet3plus", "fnet", "fnet_attn"]:
        TardisError(
            "141",
            "tardis_em/cnn/cnn.py",
            f"Wrong CNN network name {network_type}",
        )

    if network_type == "unet":
        return UNet(
            in_channels=structure["in_channel"],
            out_channels=structure["out_channel"],
            img_patch_size=img_size,
            dropout=structure["dropout"],
            conv_kernel=structure["conv_kernel"],
            padding=structure["conv_padding"],
            pool_kernel=structure["maxpool_kernel"],
            num_group=structure["num_group"],
            num_conv_layer=structure["num_conv_layers"],
            conv_layer_scaler=structure["conv_scaler"],
            layer_components=structure["layer_components"],
            prediction=prediction,
        )
    elif network_type == "resunet":
        return ResUNet(
            in_channels=structure["in_channel"],
            out_channels=structure["out_channel"],
            img_patch_size=img_size,
            dropout=structure["dropout"],
            num_conv_layer=structure["num_conv_layers"],
            conv_layer_scaler=structure["conv_scaler"],
            conv_kernel=structure["conv_kernel"],
            padding=structure["conv_padding"],
            pool_kernel=structure["maxpool_kernel"],
            num_group=structure["num_group"],
            layer_components=structure["layer_components"],
            prediction=prediction,
        )
    elif network_type == "unet3plus":
        return UNet3Plus(
            in_channels=structure["in_channel"],
            out_channels=structure["out_channel"],
            classifies=structure["classification"],
            dropout=structure["dropout"],
            img_patch_size=img_size,
            conv_kernel=structure["conv_kernel"],
            padding=structure["conv_padding"],
            pool_kernel=structure["maxpool_kernel"],
            num_conv_layer=structure["num_conv_layers"],
            conv_layer_scaler=structure["conv_scaler"],
            layer_components=structure["layer_components"],
            num_group=structure["num_group"],
            prediction=prediction,
        )
    elif network_type == "fnet":
        return FNet(
            in_channels=structure["in_channel"],
            out_channels=structure["out_channel"],
            img_patch_size=img_size,
            dropout=structure["dropout"],
            conv_kernel=structure["conv_kernel"],
            padding=structure["conv_padding"],
            pool_kernel=structure["maxpool_kernel"],
            num_conv_layer=structure["num_conv_layers"],
            conv_layer_scaler=structure["conv_scaler"],
            layer_components=structure["layer_components"],
            num_group=structure["num_group"],
            prediction=prediction,
            attn_features=False,
        )
    elif network_type == "fnet_attn":
        return FNet(
            in_channels=structure["in_channel"],
            out_channels=structure["out_channel"],
            img_patch_size=img_size,
            dropout=structure["dropout"],
            conv_kernel=structure["conv_kernel"],
            padding=structure["conv_padding"],
            pool_kernel=structure["maxpool_kernel"],
            num_conv_layer=structure["num_conv_layers"],
            conv_layer_scaler=structure["conv_scaler"],
            layer_components=structure["layer_components"],
            num_group=structure["num_group"],
            prediction=prediction,
            attn_features=True,
        )
    else:
        return None
