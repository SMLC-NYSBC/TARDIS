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

from tardis_em.cnn.model.convolution import (
    DoubleConvolution,
    RecurrentDoubleConvolution,
)
from tardis_em.cnn.model.init_weights import init_weights
from tardis_em.cnn.utils.utils import number_of_features_per_level


class DecoderBlockCNN(nn.Module):
    """
    CNN DECODER BUILDER FOR UNET

    Create a decoder block consisting of indicated number of deconvolutions followed
    by up-sampling and concatenation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        size (int): Size for the resampling.
        dropout (float, optional): Optional, dropout rate.
        components (str): String list of components from which convolution block
            is composed
        num_group (int): No. of groups for nn.GroupNorm()
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        conv_kernel: int,
        padding: int,
        size: int,
        dropout: Optional[float] = None,
        components="3gcr",
        num_group=8,
    ):
        super(DecoderBlockCNN, self).__init__()

        self.dropout = dropout

        """Build decoders"""
        if "3" in components:
            self.upscale = nn.Upsample(size=size, mode="trilinear", align_corners=False)
        elif "2" in components:
            self.upscale = nn.Upsample(size=size, mode="bilinear", align_corners=False)

        self.deconv_module = DoubleConvolution(
            in_ch=in_ch,
            out_ch=out_ch,
            block_type="decoder",
            kernel=conv_kernel,
            padding=padding,
            components=components,
            num_group=num_group,
        )

        """Optional Dropout"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        """Initialise the blocks"""
        for m in self.children():
            if m.__class__.__name__.find("Conv3d") != -1:
                continue
            if m.__class__.__name__.find("Conv2d") != -1:
                continue
            if m.__class__.__name__.find("GroupNorm") != -1:
                continue

            init_weights(m)

    def forward(self, encoder_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward CNN decoder block for Unet

        Args:
            encoder_features (torch.Tensor): Residual connection.
            x (torch.Tensor): Image tensor before convolution.

        Returns:
            torch.Tensor: Image tensor after convolution.
        """

        x = encoder_features + self.deconv_module(self.upscale(x))

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


class DecoderBlockRCNN(nn.Module):
    """
    RCNN DECODER BUILDER

    Create a decoder block consisting of indicated number of deconvolutions followed
    by upscale and connection with torch.cat().

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        size (int): Size for the resampling.
        dropout (float, optional): Optional, dropout rate.
        components (str): String list of components from which convolution block
            is composed
        num_group (int): No. of groups for nn.GroupNorm()
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        conv_kernel: int,
        padding: int,
        size: int,
        dropout: Optional[float] = None,
        components="3gcr",
        num_group=8,
    ):
        super(DecoderBlockRCNN, self).__init__()

        self.dropout = dropout

        """Build decoders"""
        if "3" in components:
            self.upscale = nn.Upsample(size=size, mode="trilinear", align_corners=False)
        elif "2" in components:
            self.upscale = nn.Upsample(size=size, mode="bilinear", align_corners=False)

        self.deconv_module = DoubleConvolution(
            in_ch=in_ch,
            out_ch=out_ch,
            block_type="decoder",
            kernel=conv_kernel,
            padding=padding,
            components=components,
            num_group=num_group,
        )

        self.deconv_res_module = RecurrentDoubleConvolution(
            in_ch=out_ch,
            out_ch=out_ch,
            block_type="decoder",
            kernel=conv_kernel,
            padding=padding,
            components=components,
            num_group=num_group,
        )

        """Optional Dropout"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        """Initialise the blocks"""
        for m in self.children():
            if m.__class__.__name__.find("Conv3d") != -1:
                continue
            if m.__class__.__name__.find("Conv2d") != -1:
                continue
            if m.__class__.__name__.find("GroupNorm") != -1:
                continue

            init_weights(m)

    def forward(self, encoder_features: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward RCNN decoder block

        Args:
            encoder_features (torch.Tensor): Residual connection.
            x (torch.Tensor): Image tensor before convolution.

        Returns:
            torch.Tensor: Image tensor after convolution.
        """
        x = self.upscale(x)
        x = self.deconv_module(x)

        x = encoder_features + x
        x = self.deconv_res_module(x)

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


class DecoderBlockUnet3Plus(nn.Module):
    """
    CNN DECODER BUILDER FOR UNET3PLUS

    Create a decoder block consisting of the indicated number of deconvolutions followed
    by upscale and connection with torch.cat().

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        size (int): Size for the resampling.
        dropout (float, optional): Optional, dropout rate.
        components (str): String list of components from which convolution block
            is composed.
        num_group (int): Number of groups inc CNN block.
        num_layer (int): Number of CNN layers.
        encoder_feature_ch (list): List of encode outputs.
        num_group (int): No. of groups for nn.GroupNorm()
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        conv_kernel: int,
        padding: int,
        size: int,
        components: str,
        num_group: int,
        num_layer: int,
        encoder_feature_ch: list,
        attn_features=False,
        dropout: Optional[float] = None,
    ):
        super(DecoderBlockUnet3Plus, self).__init__()

        self.attn_features = attn_features
        self.dropout = dropout

        """Main Block Up-Convolution"""
        if "3" in components:
            self.upscale = nn.Upsample(size=size, mode="trilinear", align_corners=False)
        elif "2" in components:
            self.upscale = nn.Upsample(size=size, mode="bilinear", align_corners=False)

        self.deconv = DoubleConvolution(
            in_ch=in_ch,
            out_ch=out_ch,
            block_type="decoder",
            kernel=conv_kernel,
            padding=padding,
            components=components,
            num_group=num_group,
        )

        """Skip-Connection Encoders"""
        num_layer = num_layer - 1
        pool_kernels = [None, 2, 4, 8, 16, 32]
        pool_kernels = pool_kernels[:num_layer]

        self.encoder_max_pool = nn.ModuleList([])
        self.encoder_feature_conv = nn.ModuleList([])

        for i, en_in_channel in enumerate(encoder_feature_ch):
            pool_kernel = pool_kernels[i]

            if pool_kernel is not None and "3" in components:
                max_pool = nn.MaxPool3d(
                    kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True
                )
            elif pool_kernel is not None and "2" in components:
                max_pool = nn.MaxPool2d(
                    kernel_size=pool_kernel, stride=pool_kernel, ceil_mode=True
                )
            else:
                max_pool = None

            conv = DoubleConvolution(
                in_ch=en_in_channel,
                out_ch=out_ch,
                block_type="decoder",
                kernel=conv_kernel,
                padding=padding,
                components=components,
                num_group=num_group,
            )
            self.encoder_max_pool.append(max_pool)
            self.encoder_feature_conv.append(conv)

        if attn_features is not None:
            self.attn_conv = DoubleConvolution(
                in_ch=out_ch + in_ch,
                out_ch=out_ch,
                block_type="decoder",
                kernel=conv_kernel,
                padding=padding,
                components=components,
                num_group=num_group,
            )

        """Optional Dropout"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        """Initialise the blocks"""
        for m in self.children():
            if m.__class__.__name__.find("Conv3d") != -1:
                continue
            if m.__class__.__name__.find("Conv2d") != -1:
                continue
            if m.__class__.__name__.find("GroupNorm") != -1:
                continue

            init_weights(m)

    def forward(self, x: torch.Tensor, encoder_features: list) -> torch.Tensor:
        """
        Forward CNN decoder block for Unet3Plus

        Args:
            x (torch.Tensor): Image tensor before convolution.
            encoder_features (list): Residual connection from the encoder.

        Returns:
            torch.Tensor: Image tensor after convolution.
        """

        """Main Block"""
        if self.attn_features:
            x_attn = self.upscale(x)
            x = self.deconv(x_attn)
        else:
            x = self.deconv(self.upscale(x))

        """Skip-Connections Encoder"""
        for i, encoder in enumerate(encoder_features):
            if self.encoder_max_pool[i] is not None:
                encoder = self.encoder_feature_conv[i](
                    self.encoder_max_pool[i](encoder)
                )
            else:
                encoder = self.encoder_feature_conv[i](encoder)
            x = x + encoder

        if self.attn_features:
            x = self.attn_conv(torch.cat((x, x_attn), dim=1))

        # Additional Dropout
        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


def build_decoder(
    conv_layers: int,
    conv_layer_scaler: int,
    components: str,
    num_group: int,
    conv_kernel: int,
    padding: int,
    sizes: list,
    dropout: Optional[float] = None,
    deconv_module="CNN",
    attn_features=False,
):
    """
    Decoder wrapper for the entire CNN model.

    Create a decoder block from a number of convolution and convolution modules.
    The Decoder block is followed by upscale(interpolation) and joining with
    torch.cat()

    Args:
        conv_layers (int): Number of deconvolution layers.
        conv_layer_scaler (int): Number of channels by which each CNN block is scaled up.
        components (str): Components that are used for deconvolution block.
        conv_kernel (int): Convolution kernel size.
        padding (int): Padding size for the convolution.
        sizes (list): List of tensor sizes for upscale.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.
        dropout (float, Optional): Dropout value.
        deconv_module: Module of the deconvolution for the decoder.
        attn_features:

    Returns:
        nn.ModuleList: List of decoder blocks.
    """
    decoders = []
    feature_map = number_of_features_per_level(
        channel_scaler=conv_layer_scaler, num_levels=conv_layers
    )
    feature_map = list(reversed(feature_map))

    """Forward decoder"""
    if deconv_module == "CNN":
        # Unet decoder
        for i in range(len(feature_map) - 1):
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            decoder = DecoderBlockCNN(
                in_ch=in_ch,
                out_ch=out_ch,
                conv_kernel=conv_kernel,
                padding=padding,
                size=size,
                dropout=dropout,
                components=components,
                num_group=num_group,
            )
            decoders.append(decoder)
    elif deconv_module == "RCNN":
        # ResNet decoder
        for i in range(len(feature_map) - 1):
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            decoder = DecoderBlockRCNN(
                in_ch=in_ch,
                out_ch=out_ch,
                conv_kernel=conv_kernel,
                padding=padding,
                size=size,
                dropout=dropout,
                components=components,
                num_group=num_group,
            )

            decoders.append(decoder)
    elif deconv_module == "unet3plus":
        # Unet3Plus decoder
        idx_en = 1

        for i in range(len(feature_map) - 1):
            # Main Module features
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            # Encoder De Convolution
            encoder_feature_ch = feature_map[idx_en:]
            idx_en += 1

            decoder = DecoderBlockUnet3Plus(
                in_ch=in_ch,
                out_ch=out_ch,
                conv_kernel=conv_kernel,
                padding=padding,
                size=size,
                components=components,
                num_group=num_group,
                num_layer=conv_layers,
                encoder_feature_ch=encoder_feature_ch,
                attn_features=attn_features,
                dropout=dropout,
            )
            decoders.append(decoder)

    return nn.ModuleList(decoders)
