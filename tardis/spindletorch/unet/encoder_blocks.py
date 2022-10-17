from typing import Optional

import torch.nn as nn
# from tardis.spindletorch.unet.init_weights import init_weights
from tardis.spindletorch.utils.utils import number_of_features_per_level


class EncoderBlock(nn.Module):
    """
    Single encoder module composed of nn.MaxPool3d and
    convolution module.

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        conv_module: Single of Double convolution module
        conv_kernel: Convolution kernel size
        max_pool: If True nn.MaxPool3d is applied
        pool_kernel: Kernel size for max pooling
        padding: Additional padding used for convolution
        conv_component: Components used to build conv_module
        no_groups: number of group used to nn.GroupNorm
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 conv_module,
                 conv_kernel=3,
                 max_pool=True,
                 dropout: Optional[float] = None,
                 pool_kernel=2,
                 padding=1,
                 conv_component="3gcr",
                 no_groups=8):
        super(EncoderBlock, self).__init__()
        self.dropout = dropout

        if max_pool:
            if '3' in conv_component:
                self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel)
            elif '2' in conv_component:
                self.max_pool = nn.MaxPool2d(kernel_size=pool_kernel)
        else:
            self.max_pool = None

        if dropout is not None:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_module = conv_module(in_ch=in_ch,
                                       out_ch=out_ch,
                                       block_type="encoder",
                                       kernel=conv_kernel,
                                       padding=padding,
                                       components=conv_component,
                                       no_group=no_groups)

 

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)

        x = self.conv_module(x)

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


def build_encoder(in_ch: int,
                  conv_layers: int,
                  conv_layer_multiplayer: int,
                  conv_kernel: int or tuple,
                  padding: int or tuple,
                  no_groups: int,
                  components: str,
                  pool_kernel: int or tuple,
                  conv_module,
                  dropout: Optional[float] = None):
    """
    Create encoder block from feature map and convolution modules. Number of
    encoder layers is indicated by number of features

    Args:
        in_ch: Number of input channels
        conv_layers: Number of convolution layers (each layer is scaled by 64)
        conv_module: Module of convolution for encoder
        conv_kernel: Kernel size for convolution
        padding: Padding size for convolution
        no_groups: No. of groups for nn.GroupNorm
        components: String list of components for convolution block
        pool_kernel: Kernel size for nn.MaxPool3d()

    Returns: nn.ModuleList
    """
    encoders = []
    feature_map = number_of_features_per_level(conv_layer_multiplayer,
                                               conv_layers)

    for i, feature in enumerate(feature_map):
        if i == 0:  # first encoder layer skips max pooling
            encoder = EncoderBlock(in_ch=in_ch,
                                   out_ch=feature,
                                   conv_module=conv_module,
                                   conv_kernel=conv_kernel,
                                   dropout=dropout,
                                   max_pool=False,
                                   padding=padding,
                                   conv_component=components,
                                   no_groups=no_groups)
        else:
            encoder = EncoderBlock(in_ch=feature_map[i - 1],
                                   out_ch=feature,
                                   conv_module=conv_module,
                                   conv_kernel=conv_kernel,
                                   dropout=dropout,
                                   max_pool=True,
                                   pool_kernel=pool_kernel,
                                   padding=padding,
                                   conv_component=components,
                                   no_groups=no_groups)
        encoders.append(encoder)

    return nn.ModuleList(encoders)
