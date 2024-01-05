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
import torch.nn.functional as F

from tardis_em.cnn.model.init_weights import init_weights
from tardis_em.cnn.utils.utils import number_of_features_per_level
from tardis_em.cnn.model.convolution import SingleConvolution


class EncoderBlock(nn.Module):
    """
    ENCODER BUILDER

    Single encoder module composed of nn.MaxPool and convolution module.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        conv_module (conv_module): Single, Double or RCNN convolution block.
        conv_kernel (int): Convolution kernel size.
        max_pool (int): If True nn.MaxPool is applied.
        pool_kernel (int): Kernel size for max pooling.
        dropout (float, optional): Optionals, dropout rate.
        padding (int): Padding size for the convolution.
        components (str): Components that are used for conv. block.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.
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
        multi_res=False,
        skip_connections=False,
    ):
        super(EncoderBlock, self).__init__()
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

        if multi_res:
            if "3" in components:
                self.init_embedding = EncoderEmbedding(components='3gc')
            else:
                self.init_embedding = EncoderEmbedding(components='2gc')
        else:
            self.init_embedding = None

        """Initialise the blocks"""
        for m in self.children():
            init_weights(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward CNN encoder block.

        Args:
            x (torch.Tensor): Image torch before convolution.

        Returns:
            torch.Tensor: Image after convolution.
        """
        if self.init_embedding is not None:
            x = self.init_embedding(x)

        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.conv_module(x)

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


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
    multi_res = False,
    skip_connections=False,
) -> nn.ModuleList:
    """
    Encoder wrapper for entire CNN model.

    Create encoder block from feature map and convolution modules. Number of
    encoder layers is indicated by number of features.

    Args:
        in_ch (int): Number of input channels.
        conv_layers (int): Number of convolution layers.
        conv_layer_scaler (int): Number of channel by which each CNN block is scaled up.
        conv_module (conv_module): Single, Double or RCNN convolution block.
        conv_kernel (int): Convolution kernel size.
        pool_kernel (int): Kernel size for max pooling.
        dropout (float, optional): Optionals, dropout rate.
        padding (int): Padding size for the convolution.
        components (str): Components that are used for conv. block.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.

    Returns:
        nn.ModuleList: Encoder block.
    """
    if multi_res:
        in_ch = 5

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
                multi_res=multi_res,
                skip_connections=False,
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
                skip_connections=skip_connections,
            )
        encoders.append(encoder)

    return nn.ModuleList(encoders)


class EncoderEmbedding(nn.Module):
    def __init__(self,
                 components='3gc'):
        super(EncoderEmbedding, self).__init__()
        
        """Resolution embedding"""
        self.conv_2x = SingleConvolution(1, 1, components, 3, 1, 8, 'encoder')
        self.conv_1x = SingleConvolution(1, 1, components, 3, 1, 8, 'encoder')
        self.conv_05x = SingleConvolution(1, 1, components, 3, 1, 8, 'encoder')
        self.fc = SingleConvolution(1, 1, components, 3, 1, 8, 'encoder')
        
        """Initialise the blocks"""
        for m in self.children():
            init_weights(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        """2x - Resolution"""
        x2 = F.interpolate(x, scale_factor=2, mode='trilinear')
        x2 = self.conv_2x(x2)
        x2 = F.interpolate(x2, scale_factor=0.5, mode='trilinear')
        
        """1x - Resolution"""
        x1 = self.conv_1x(x)
    
        """1/2x - Resolution"""
        x05 = F.interpolate(x, scale_factor=2, mode='trilinear')
        x05 = self.conv_2x(x05)
        x05 = F.interpolate(x05, scale_factor=0.5, mode='trilinear')
    
        x_embed = self.fc(x2 + x1 + x05)

        return torch.cat([x, x2, x1, x05, x_embed], dim=1)
