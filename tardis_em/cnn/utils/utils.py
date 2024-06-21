#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import numpy as np


def number_of_features_per_level(channel_scaler: int, num_levels: int) -> list:
    """
    Compute list of output channels for CNN.

    Features = channel_scaler * 2^k
        where:
        - k is layer number

    Args:
        channel_scaler (int): Number of initial input channels for CNN.
        num_levels (int): Number of channels from max_number_of_conv_layer().

    Returns:
        list: List of output channels.
    """
    return [channel_scaler * 2**k for k in range(num_levels)]


def max_number_of_conv_layer(
    img=None,
    input_volume=64,
    max_out=8,
    kernel_size=3,
    padding=1,
    stride=1,
    pool_size=2,
    pool_stride=2,
    first_max_pool=False,
) -> int:
    """
    Calculate maximum possible number of layers given image size.

    Based on the torch input automatically select number of convolution blocks,
    based on a standard settings.

        I = [(W - K + 2*P) / S] + 1
        Out_size = [(I - F) / S] + 1
        - W is the input volume
        - K is the Kernel size
        - P is the padding
        - S is the stride
        - F is the max pooling kernel size

    Args:
        img (np.ndarray, Optional): Tensor data from which size of is calculated.
        input_volume (int): Size of the multiplayer for the convolution.
        max_out (int): Maximal output dimension after max_pooling.
        kernel_size (int): Kernel size for the convolution.
        padding (int): Padding size for the convolution blocks.
        stride: (int) Stride for the convolution blocks.
        pool_size (int) Max Pooling kernel size.
        pool_stride (int): Max Pooling stride size.
        first_max_pool (bool): If first CNN block has max pooling.

    Returns:
        int: Maximum number of CNN layers.
    """
    if img is not None:
        # In case of anisotropic image select smallest size
        # to calculate max conv layers
        if len(img.shape) == 5:
            size = np.array(list(img.shape[2:]))
        elif len(img.shape) == 4:
            size = np.array(list(img.shape[1:]))
        else:
            size = np.array(list(img.shape))

        size = size[size.argmin()]
        input_volume = size

    max_layers = 0

    while input_volume >= max_out:
        img_window = ((input_volume - kernel_size + 2 * padding) / stride) + 1
        input_volume = ((img_window - pool_size) / pool_stride) + 1
        max_layers += 1

    if first_max_pool:
        max_layers -= 1

    return max_layers


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Simple image data normalizer between 0,1.

    Args:
        image: Image data set.

    Returns:
        np.ndarray: Normalized image between 0 and 1 values.
    """
    image_min = np.min(image)
    image_max = np.max(image)

    if image_min == 0 and image_max == 1:
        return image

    if image_min == 0:
        image = np.where(image > image_min, 1, 0)
    elif image_max == 0:
        image = np.where(image < image_max, 1, 0)

    return image


def check_model_dict(model_dict: dict) -> dict:
    """
    Check and rebuild model structure dictionary to ensure back-compatibility.

    Args:
        model_dict (dict): Model structure dictionary.

    Returns:
        dict: Standardize model structure dictionary.
    """
    new_dict = {}

    for key, value in model_dict.items():
        if key.endswith("type"):
            new_dict["cnn_type"] = value
        if key.endswith("cation"):
            new_dict["classification"] = value
        if key.endswith("_in") or key.startswith("in_"):
            new_dict["in_channel"] = value
        if key.endswith("_out") or key.startswith("out_"):
            new_dict["out_channel"] = value
        if key.endswith("size"):
            new_dict["img_size"] = value
        if key.endswith("dropout"):
            new_dict["dropout"] = value
        if key.endswith("layers"):
            new_dict["num_conv_layers"] = value
        if key.endswith("scaler") or key.endswith("multiplayer"):
            new_dict["conv_scaler"] = value
        if key.endswith("v_kernel"):
            new_dict["conv_kernel"] = value
        if key.endswith("padding"):
            new_dict["conv_padding"] = value
        if key.endswith("l_kernel"):
            new_dict["maxpool_kernel"] = value
        if key.endswith("components"):
            new_dict["layer_components"] = value
        if key.endswith("features"):
            new_dict["attn_features"] = value
        if key.endswith("group"):
            new_dict["num_group"] = value

    return new_dict
