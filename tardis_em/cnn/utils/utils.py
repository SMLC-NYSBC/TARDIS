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
    Calculates the number of features per level in a hierarchical structure, where the feature count
    doubles at each subsequent level starting from the initial level. The initial feature count is
    specified by `channel_scaler`, and the total number of levels is defined by `num_levels`.

    :param channel_scaler: Initial feature count at the first level.
    :type channel_scaler: int
    :param num_levels: Total number of levels in the hierarchical structure.
    :type num_levels: int

    :return: A list containing the feature counts for each level, starting from the
             initial level up to the final level.
    :rtype: list
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
    Calculates the maximum possible number of convolutional layers in a neural
    network, given specific image dimensions, convolution parameters, and pooling
    parameters. If an image is provided, the calculation uses the smallest
    dimension of the image for the computation.

    :param img: An optional input tensor of image data, which may be a 4D
        (batch_size, channels, height, width) or 5D (batch_size, time, channels,
        height, width) tensor. If provided, the computational dimension will be
        chosen based on the smallest spatial value.
    :param input_volume: The input volume size (spatial dimension) for the
        computation, if no image is provided. This is the starting dimension for
        estimating possible convolutional layers.
    :param max_out: The minimum allowable output volume size that determines when
        computation for convolutional layers should end.
    :param kernel_size: The size of the convolutional kernel/filter.
    :param padding: The number of zero-padding pixels added around each
        convolutional operation.
    :param stride: The stride size for the convolutional operation.
    :param pool_size: The size of the pooling window for down-sampling.
    :param pool_stride: The stride size for pooling.
    :param first_max_pool: A flag determining whether the first max-pooling
        operation should be applied. If set to True, the number of layers is
        reduced by one to account for the absence of a pooling step.

    :return: The maximum possible number of convolutional layers as an integer
        that can fit within the specified dimensions and parameters.
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
    Normalizes the given image by checking its minimum and maximum values. The
    function ensures that the image is either normalized between 0 and 1 or
    converted into a binary representation based on the pixel intensity.

    :param image: An array representing the image to be normalized.
    :type image: np.ndarray

    :return: A normalized image array.
    :rtype: np.ndarray
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
    Analyzes the provided dictionary containing model configuration and maps its keys to
    a standardized format for further processing. This function ensures that specific key
    patterns in the input dictionary are identified and their corresponding values are
    transferred into a new dictionary with standardized key names.

    :param model_dict: The input dictionary containing model configuration settings.
                       Keys may represent various model attributes and must conform to
                       specific naming patterns to be mapped accordingly.
    :type model_dict: dict

    :return: A new dictionary containing remapped key-value pairs, where keys follow a
             standardized naming convention. Only keys matching the predefined patterns
             are included in the output dictionary.
    :rtype: dict
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
