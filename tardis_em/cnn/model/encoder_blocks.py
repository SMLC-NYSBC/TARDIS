#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional

import torch
import torch.nn as nn

from tardis_em.cnn.model.init_weights import init_weights
from tardis_em.cnn.utils.utils import number_of_features_per_level


class EncoderBlock(nn.Module):
    """
    Represents an encoder block with convolutional modules, optional max-pooling, dropout, and
    attention features for deep learning architectures.

    This class constructs a configurable encoder block. The encoder block supports max-pooling
    layers, dropout layers, attention features, and employs convolutional modules with specific
    kernel sizes and padding. The components used in the block can be customized through
    component identifiers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        conv_module,
        conv_kernel=3,
        max_pool=True,
        dropout: Optional[float] = None,
        pool_kernel=2,
        padding=1,
        components="3gcr",
        num_group=8,
        attn_features=False,
    ):
        """
        Initializes an encoder block with optional max pooling, dropout, and convolutional
        modules. The encoder block can also support attention features when enabled. The
        constructor sets up the required modules, components, and parameters based on the
        provided input arguments.

        :param in_ch: Number of input channels for this encoder block.
        :type in_ch: int
        :param out_ch: Number of output channels for this encoder block.
        :type out_ch: int
        :param conv_module: A convolutional module or block used within the encoder.
        :param conv_kernel: Size of the convolutional kernel (default is 3).
        :type conv_kernel: int, optional
        :param max_pool: If True, adds a max-pooling layer to the block (default is True).
        :type max_pool: bool, optional
        :param dropout: Dropout rate to be applied. If None, no dropout layer is added.
        :type dropout: float, optional
        :param pool_kernel: Size of the kernel for max-pooling operations (default is 2).
        :type pool_kernel: int, optional
        :param padding: Amount of padding to apply in convolutional layers (default is 1).
        :type padding: int, optional
        :param components: String identifier defining specific layer components to use
            (e.g., "3gcr").
        :type components: str
        :param num_group: Number of groups for grouped convolutions (default is 8).
        :type num_group: int, optional
        :param attn_features: Enables the attention feature if set to True (default is False).
        :type attn_features: bool, optional

        :raises TypeError: If `conv_module` is not callable.
        :raises ValueError: If incorrect component identifiers are provided in `components` or
            if arguments are incompatible with the selected configuration.
        """
        super(EncoderBlock, self).__init__()

        self.attn_features = attn_features
        self.dropout = dropout

        """Optionally, add maxpool"""
        if max_pool:
            if "3" in components:
                self.maxpool = nn.MaxPool3d(kernel_size=pool_kernel)
            elif "2" in components:
                self.maxpool = nn.MaxPool2d(kernel_size=pool_kernel)
        else:
            self.maxpool = None

        """Optionally, add dropout layer"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(p=dropout)

        """Build CNN block"""
        self.conv_module = conv_module(
            in_ch=in_ch,
            out_ch=out_ch,
            block_type="encoder",
            kernel=conv_kernel,
            padding=padding,
            components=components,
            num_group=num_group,
        )

        if self.attn_features:
            self.attn_conv = conv_module(
                in_ch=in_ch + out_ch,
                out_ch=out_ch,
                block_type="encoder",
                kernel=conv_kernel,
                padding=padding,
                components=components,
                num_group=1 if (in_ch + out_ch) % num_group != 0 else num_group,
            )

        """Initialise the blocks"""
        for m in self.children():
            init_weights(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes input tensor through layers including convolutional, attention mechanisms,
        optional pooling, and dropout, returning the transformed tensor. This method serves as
        the key forward computation logic for the module.

        :param x: Input tensor to be processed.
        :type x: torch.Tensor

        :return: Processed output tensor.
        :rtype: torch.Tensor
        """
        if self.maxpool is not None:
            x = self.maxpool(x)

        x_attn = self.conv_module(x)

        if self.attn_features:
            x_attn = self.attn_conv(torch.cat((x, x_attn), dim=1))

        if self.dropout is not None:
            x_attn = self.dropout_layer(x_attn)

        return x_attn


def build_encoder(
    in_ch: int,
    conv_layers: int,
    conv_layer_scaler: int,
    conv_kernel: int or tuple,
    padding: int or tuple,
    num_group: int,
    components: str,
    pool_kernel: int or tuple,
    conv_module,
    dropout: Optional[float] = None,
    attn_features=False,
) -> nn.ModuleList:
    """
    Constructs and returns a sequence of encoder blocks as a module list. Each encoder block
    is defined based on the provided parameters and forms a hierarchical structure of
    feature extraction layers. The first encoder block does not use max pooling, while
    subsequent layers include max pooling operations as specified. The method also
    accommodates advanced features such as attention mechanisms and grouped convolutions.

    :param in_ch: Number of input image channels for the first encoder block. Each subsequent
        block adapts based on feature scaling.
    :param conv_layers: Number of convolutional layers to create in the encoder architecture.
    :param conv_layer_scaler: Multiplier to compute the number of feature maps at each level
        of the encoder.
    :param conv_kernel: Kernel size for convolution operations in the encoder blocks. Can be
        an integer or tuple.
    :param padding: Padding value for the convolutional layers. Accepts integer or tuple for
        specific layer configurations.
    :param num_group: Number of groups for grouped convolution operations.
    :param components: String identifier or configuration for additional encoder components
        such as normalization layers or activation functions.
    :param pool_kernel: Kernel size used for max pooling operations across encoder blocks
        except the first block. Accepts integer or tuple.
    :param conv_module: A callable or module defining the specific implementation for
        convolutional operations.
    :param dropout: Optional dropout probability to apply regularization in each encoder
        block. Defaults to None, disabling dropout.
    :param attn_features: Boolean flag to enable or disable attention mechanisms in the encoder
        blocks. Defaults to False.

    :return: Returns a `nn.ModuleList` containing the constructed encoder blocks.
    """
    encoders = []
    feature_map = number_of_features_per_level(
        channel_scaler=conv_layer_scaler, num_levels=conv_layers
    )

    for i, feature in enumerate(feature_map):
        if i == 0:  # first encoder layer skips max pooling
            encoder = EncoderBlock(
                in_ch=in_ch,
                out_ch=feature,
                conv_module=conv_module,
                conv_kernel=conv_kernel,
                dropout=dropout,
                max_pool=False,
                padding=padding,
                components=components,
                num_group=num_group,
                attn_features=attn_features,
            )
        else:
            encoder = EncoderBlock(
                in_ch=feature_map[i - 1],
                out_ch=feature,
                conv_module=conv_module,
                conv_kernel=conv_kernel,
                dropout=dropout,
                pool_kernel=pool_kernel,
                padding=padding,
                components=components,
                num_group=num_group,
                attn_features=attn_features,
            )
        encoders.append(encoder)

    return nn.ModuleList(encoders)
