import numpy as np


def number_of_features_per_level(init_channel_number=64,
                                 num_levels=5):
    """
    Compute list of output_channels for CNN features

    Args:
        init_channel_number: Number of initial input channels for CNN
        num_levels: Number of channels from max_number_of_conv_layer()

    """
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def max_number_of_conv_layer(img=None,
                             input_volume=64,
                             max_out=8,
                             kernel_size=3,
                             padding=1,
                             stride=1,
                             pool_size=2,
                             pool_stride=2,
                             first_max_pool=False):
    """
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
        img: Tensor Image data if from which size of data is calculated
        input_volume: Size of the multiplayer for the convolution
        max_out: Maximal output dimension after maxpooling
        kernel_size: Kernel size for the convolution
        padding: Padding size for the convolution blocks
        stride: Stride for the convolution blocks
        pool_size: Max Pooling kernel size
        pool_stride: Max Pooling stride size
        first_max_pool: If first CNN block has max pooling
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
