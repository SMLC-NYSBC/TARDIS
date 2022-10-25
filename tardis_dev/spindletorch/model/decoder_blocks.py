from typing import Optional

import torch
import torch.nn as nn
from tardis_dev.spindletorch.model.convolution import (DoubleConvolution,
                                                       RecurrentDoubleConvolution)
from tardis_dev.spindletorch.model.init_weights import init_weights
from tardis_dev.spindletorch.utils.utils import number_of_features_per_level


class DecoderBlockCNN(nn.Module):
    """
    CNN DECODER BUILDER FOR UNET

    Create decoder block consist of indicated number of deconvolution followed
    by up-sampling and concatenation.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        size (int): Size for the resampling.
        dropout (float, optional): Optional, dropout rate.
        components: String list of components from which convolution block
            is composed
        num_group: No. of groups for nn.GroupNorm()
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 size: int,
                 dropout: Optional[float] = None,
                 components="3gcr",
                 num_group=8):
        super(DecoderBlockCNN, self).__init__()

        self.dropout = dropout

        """Build decoders"""
        if '3' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='trilinear',
                                        align_corners=False)
        elif '2' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='bilinear',
                                        align_corners=False)

        self.deconv_module = DoubleConvolution(in_ch=in_ch,
                                               out_ch=out_ch,
                                               block_type="decoder",
                                               kernel=3,
                                               padding=1,
                                               components=components,
                                               num_group=num_group)

        """Optional Dropout"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        """Initialise the blocks"""
        for m in self.children():
            if m.__class__.__name__.find('Conv3d') != -1:
                continue
            if m.__class__.__name__.find('Conv2d') != -1:
                continue
            if m.__class__.__name__.find('GroupNorm') != -1:
                continue

            init_weights(m, init_type='kaiming')

    def forward(self,
                encoder_features: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward CNN decoder block for Unet

        Args:
            encoder_features (torch.Tensor): Residual connection.
            x (torch.Tensor): Image tensor before convolution.

        Returns:
            torch.Tensor: Image tensor after convolution.
        """
        x = self.upsample(x)
        x = self.deconv_module(x)
        x = encoder_features + x

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


class DecoderBlockRCNN(nn.Module):
    """
    RCNN DECODER BUILDER

    Create decoder block consist of indicated number of deconvolution followed
    by upsampling and connection with torch.cat().

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        size (int): Size for the resampling.
        dropout (float, optional): Optional, dropout rate.
        components: String list of components from which convolution block
            is composed
        num_group: No. of groups for nn.GroupNorm()
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 size: int,
                 dropout: Optional[float] = None,
                 components="3gcr",
                 num_group=8):
        super(DecoderBlockRCNN, self).__init__()

        self.dropout = dropout

        """Build decoders"""
        if '3' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='trilinear',
                                        align_corners=False)
        elif '2' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='bilinear',
                                        align_corners=False)

        self.deconv_module = DoubleConvolution(in_ch=in_ch,
                                               out_ch=out_ch,
                                               block_type="decoder",
                                               kernel=3,
                                               padding=1,
                                               components=components,
                                               num_group=num_group)

        self.deconv_res_module = RecurrentDoubleConvolution(in_ch=out_ch,
                                                            out_ch=out_ch,
                                                            block_type="decoder",
                                                            kernel=3,
                                                            padding=1,
                                                            components=components,
                                                            num_group=num_group)

        """Optional Dropout"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        """Initialise the blocks"""
        for m in self.children():
            if m.__class__.__name__.find('Conv3d') != -1:
                continue
            if m.__class__.__name__.find('Conv2d') != -1:
                continue
            if m.__class__.__name__.find('GroupNorm') != -1:
                continue

            init_weights(m, init_type='kaiming')

    def forward(self,
                encoder_features: torch.Tensor,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward RCNN decoder block

        Args:
            encoder_features (torch.Tensor): Residual connection.
            x (torch.Tensor): Image tensor before convolution.

        Returns:
            torch.Tensor: Image tensor after convolution.
        """
        x = self.upsample(x)
        x = self.deconv_module(x)

        x = encoder_features + x
        x = self.deconv_res_module(x)

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


class DecoderBlockUnet3Plus(nn.Module):
    """
    CNN DECODER BUILDER FOR UNET3PLUS

    Create decoder block consist of indicated number of deconvolution followed
    by upsampling and connection with torch.cat().

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        size (int): Size for the resampling.
        dropout (float, optional): Optional, dropout rate.
        components: String list of components from which convolution block
            is composed.
        num_group (int): Number ofr groups inc CNN block.
        num_layers (int): Number of CNN layers.
        decoder_feature_ch (list): List of decoder outputs.
        encoder_feature_ch (list): List of encode outputs.
        num_group: No. of groups for nn.GroupNorm()
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 size: int,
                 components: str,
                 num_group: int,
                 num_layer: int,
                 decoder_feature_ch: list,
                 encoder_feature_ch: list,
                 dropout: Optional[float] = None):
        super(DecoderBlockUnet3Plus, self).__init__()

        self.dropout = dropout

        """Main Block Up-Convolution"""
        if '3' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='trilinear',
                                        align_corners=False)
        elif '2' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='bilinear',
                                        align_corners=False)
        self.deconv = DoubleConvolution(in_ch=in_ch,
                                        out_ch=out_ch,
                                        block_type='decoder',
                                        kernel=3,
                                        padding=1,
                                        components=components,
                                        num_group=num_group)

        """Skip-Connection Encoders"""
        num_layer = num_layer - 1
        pool_kernels = [None, 2, 4, 8, 16, 32]
        pool_kernels = pool_kernels[:num_layer]

        self.encoder_max_pool = nn.ModuleList([])
        self.encoder_feature_conv = nn.ModuleList([])
        for i, en_in_channel in enumerate(encoder_feature_ch):
            pool_kernel = pool_kernels[i]

            if pool_kernel is not None and '3' in components:
                max_pool = nn.MaxPool3d(kernel_size=pool_kernel,
                                        stride=pool_kernel,
                                        dilation=1,
                                        ceil_mode=True)
            elif pool_kernel is not None and '2' in components:
                max_pool = nn.MaxPool2d(kernel_size=pool_kernel,
                                        stride=pool_kernel,
                                        dilation=1,
                                        ceil_mode=True)
            else:
                max_pool = None

            conv = DoubleConvolution(in_ch=en_in_channel,
                                     out_ch=out_ch,
                                     block_type="decoder",
                                     kernel=3,
                                     padding=1,
                                     components=components,
                                     num_group=num_group)

            self.encoder_max_pool.append(max_pool)
            self.encoder_feature_conv.append(conv)

        """Skip-Connection Decoders"""
        self.decoder_feature_upsample = nn.ModuleList([])
        self.decoder_feature_conv = nn.ModuleList([])
        for de_in_channel in decoder_feature_ch:
            if '3' in components:
                upsample = nn.Upsample(size=size,
                                       mode='trilinear',
                                       align_corners=False)
            elif '2' in components:
                upsample = nn.Upsample(size=size,
                                       mode='bilinear',
                                       align_corners=False)

            deconv_module = DoubleConvolution(in_ch=de_in_channel,
                                              out_ch=out_ch,
                                              block_type="decoder",
                                              kernel=3,
                                              padding=1,
                                              components=components,
                                              num_group=num_group)
            self.decoder_feature_upsample.append(upsample)
            self.decoder_feature_conv.append(deconv_module)

        """Optional Dropout"""
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        """Initialise the blocks"""
        for m in self.children():
            if m.__class__.__name__.find('Conv3d') != -1:
                continue
            if m.__class__.__name__.find('Conv2d') != -1:
                continue
            if m.__class__.__name__.find('GroupNorm') != -1:
                continue

            init_weights(m, init_type='kaiming')

    def forward(self,
                x: torch.Tensor,
                decoder_features: list,
                encoder_features: list) -> torch.Tensor:
        """
        Forward CNN decoder block for Unet3Plus

        Args:
            encoder_features (torch.Tensor): Residual connection.
            x (torch.Tensor): Image tensor before convolution.

        Returns:
            torch.Tensor: Image tensor after convolution.
        """

        """Main Block"""
        x = self.upsample(x)
        x = self.deconv(x)

        """Skip-Connections Encoder"""
        x_en_features = []
        for i, encoder in enumerate(encoder_features):
            max_pool_layer = self.encoder_max_pool[i]

            if max_pool_layer is not None:
                x_en = max_pool_layer(encoder)
                x_en = self.encoder_feature_conv[i](x_en)
            else:
                x_en = self.encoder_feature_conv[i](encoder)

            x_en_features.insert(0, x_en)

        if len(x_en_features) == 0:
            x_en_features = None

        """Skip-Connections Decoder"""
        x_de_features = []
        for i, decoder in enumerate(decoder_features):
            x_de = self.decoder_feature_upsample[i](decoder)
            x_de = self.decoder_feature_conv[i](x_de)
            x_de_features.insert(0, x_de)

        if len(x_de_features) == 0:
            x_de_features = None

        """Sum list of tensors"""
        if x_en_features is not None:
            for encoder in x_en_features:
                x = x + encoder

        if x_de_features is not None:
            for decoder in x_de_features:
                x = x + decoder

        # Additional Dropout
        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


def build_decoder(conv_layers: int,
                  conv_layer_scaler: int,
                  components: str,
                  num_group: int,
                  dropout: Optional[float] = None,
                  sizes=None,
                  deconv_module='CNN'):
    """
    Decoder wrapper for entire CNN model.

    Create decoder block from number of convolution and convolution modules.
    Decoder block is followed by upsampling(interpolation) and joining with
    torch.cat()

    Args:
        conv_layers: Number of deconvolution layers.
        conv_layer_scaler: Number of channel by which each CNN block is scaled up.
        conv_kernel (int): Convolution kernel size.
        padding: Padding size for the convolution.
        components: Components that are used for deconvolution block.
        num_group: Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.
        up_sampling: If True the up-sampling is applied.
        deconv_module: Module of the deconvolution for decoder.

    Returns:
        nn.ModuleList: List of decoders blocks
    """
    decoders = []
    feature_map = number_of_features_per_level(channel_scaler=conv_layer_scaler,
                                               num_levels=conv_layers)
    feature_map = list(reversed(feature_map))

    """Forward decoder"""
    if deconv_module == 'CNN':
        # Unet decoder
        for i in range(len(feature_map) - 1):
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            decoder = DecoderBlockCNN(in_ch=in_ch,
                                      out_ch=out_ch,
                                      size=size,
                                      dropout=dropout,
                                      components=components,
                                      num_group=num_group)
            decoders.append(decoder)
    elif deconv_module == 'RCNN':
        # ResNet decoder
        for i in range(len(feature_map) - 1):
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            decoder = DecoderBlockRCNN(in_ch=in_ch,
                                       out_ch=out_ch,
                                       size=size,
                                       dropout=dropout,
                                       components=components,
                                       num_group=num_group)

            decoders.append(decoder)
    elif deconv_module == 'unet3plus':
        # Unet3Plus decoder
        idx_de = len(feature_map) + 1
        idx_en = 1

        for i in range(len(feature_map) - 1):
            # Main Module features
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            # Encoder De Convolution
            encoder_feature_ch = feature_map[idx_en:]
            idx_en += 1

            # Decoder Up Convolution
            decoder_feature_ch = feature_map[idx_de:]
            idx_de -= 1

            decoder = DecoderBlockUnet3Plus(in_ch=in_ch,
                                            out_ch=out_ch,
                                            size=size,
                                            components=components,
                                            num_group=num_group,
                                            num_layer=conv_layers,
                                            encoder_feature_ch=encoder_feature_ch,
                                            decoder_feature_ch=decoder_feature_ch,
                                            dropout=dropout)
            decoders.append(decoder)

    return nn.ModuleList(decoders)
