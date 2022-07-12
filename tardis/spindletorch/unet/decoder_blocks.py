from typing import Optional

import torch
import torch.nn as nn
from tardis.spindletorch.unet.convolution import (DoubleConvolution,
                                                  RecurrentDoubleConvolution)
from tardis.spindletorch.unet.init_weights import init_weights
from tardis.spindletorch.utils.utils import number_of_features_per_level


class DecoderBlockCNN(nn.Module):
    """
    Create decoder block consist of indicated number of deconvolution followed
    by up-sampling and connection with torch.cat().

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        deconv_module: Module of deconvolution
        components: String list of components from which convolution block
            is composed
        no_group: No. of groups for nn.GroupNorm()
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 size: int,
                 dropout: Optional[float] = None,
                 components="3gcr",
                 no_group=8):
        super(DecoderBlockCNN, self).__init__()
        self.dropout = dropout

        if '3' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='trilinear',
                                        align_corners=True)
        elif '2' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='bilinear',
                                        align_corners=True)

        self.deconv_module = DoubleConvolution(in_ch=in_ch,
                                               out_ch=out_ch,
                                               block_type="decoder",
                                               kernel=3,
                                               padding=1,
                                               components=components,
                                               no_group=no_group)
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('Conv3d') != -1:
                init_weights(m, init_type='kaiming')
            if m.__class__.__name__.find('Conv2d') != -1:
                init_weights(m, init_type='kaiming')
            if m.__class__.__name__.find('GroupNorm') != -1:
                init_weights(m, init_type='kaiming')

    def forward(self,
                encoder_features: torch.Tensor,
                x: torch.Tensor):
        x = self.upsample(x)
        x = self.deconv_module(x)
        x = encoder_features + x

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


class DecoderBlockRCNN(nn.Module):
    """
    Create decoder block consist of indicated number of deconvolution followed
    by upsampling and connection with torch.cat().

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        deconv_module: Module of deconvolution
        components: String list of components from which convolution block
            is composed
        no_group: No. of groups for nn.GroupNorm()
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 size: int,
                 dropout: Optional[float] = None,
                 components="3gcr",
                 no_group=8):
        super(DecoderBlockRCNN, self).__init__()
        self.dropout = dropout

        if '3' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='trilinear',
                                        align_corners=True)
        elif '2' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='bilinear',
                                        align_corners=True)

        self.deconv_module = DoubleConvolution(in_ch=in_ch,
                                               out_ch=out_ch,
                                               block_type="decoder",
                                               kernel=3,
                                               padding=1,
                                               components=components,
                                               no_group=no_group)

        self.deconv_res_module = RecurrentDoubleConvolution(in_ch=out_ch,
                                                            out_ch=out_ch,
                                                            block_type="decoder",
                                                            kernel=3,
                                                            padding=1,
                                                            components=components,
                                                            no_group=no_group)

        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('Conv3d') != -1:
                init_weights(m, init_type='kaiming')
            if m.__class__.__name__.find('Conv2d') != -1:
                init_weights(m, init_type='kaiming')
            if m.__class__.__name__.find('GroupNorm') != -1:
                init_weights(m, init_type='kaiming')

    def forward(self,
                encoder_features: torch.Tensor,
                x: torch.Tensor):
        x = self.upsample(x)
        x = self.deconv_module(x)

        x = encoder_features + x
        x = self.deconv_res_module(x)

        if self.dropout is not None:
            x = self.dropout_layer(x)

        return x


class DecoderBlockUnet3Plus(nn.Module):
    """
    Create decoder block consist of indicated number of deconvolution followed
    by up-sampling and connection with torch.cat().

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        deconv_module: Module of deconvolution
        components: String list of components from which convolution block
            is composed
        no_group: No. of groups for nn.GroupNorm()
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 size: int,
                 components: str,
                 no_group: int,
                 no_layer: int,
                 decoder_feature_ch: list,
                 encoder_feature_ch: list,
                 dropout: Optional[float] = None):
        super(DecoderBlockUnet3Plus, self).__init__()
        self.dropout = dropout

        # Main Block Up Convolution
        if '3' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='trilinear',
                                        align_corners=True)
        elif '2' in components:
            self.upsample = nn.Upsample(size=size,
                                        mode='bilinear',
                                        align_corners=True)
        self.deconv = DoubleConvolution(in_ch=in_ch,
                                        out_ch=out_ch,
                                        block_type='decoder',
                                        kernel=3,
                                        padding=1,
                                        components=components,
                                        no_group=no_group)

        # Skip-Connection Encoder
        no_layer = no_layer - 1
        pool_kernels = [None, 2, 4, 8, 16, 32]
        pool_kernels = pool_kernels[:no_layer]

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
                                     no_group=no_group)

            self.encoder_max_pool.append(max_pool)
            self.encoder_feature_conv.append(conv)

        # Skip-Connection Decoder
        self.decoder_feature_upsample = nn.ModuleList([])
        self.decoder_feature_conv = nn.ModuleList([])
        for de_in_channel in decoder_feature_ch:
            if '3' in components:
                upsample = nn.Upsample(size=size,
                                       mode='trilinear',
                                       align_corners=True)
            elif '2' in components:
                upsample = nn.Upsample(size=size,
                                       mode='bilinear',
                                       align_corners=True)

            deconv_module = DoubleConvolution(in_ch=de_in_channel,
                                              out_ch=out_ch,
                                              block_type="decoder",
                                              kernel=3,
                                              padding=1,
                                              components=components,
                                              no_group=no_group)
            self.decoder_feature_upsample.append(upsample)
            self.decoder_feature_conv.append(deconv_module)

        # Additional Dropout
        if dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)

        # initialise the blocks
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
                encoder_features: list):

        # Main Block
        x = self.upsample(x)
        x = self.deconv(x)

        # Skip-Connections Encoder
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

        # Skip-Connections Decoder
        x_de_features = []
        for i, decoder in enumerate(decoder_features):
            x_de = self.decoder_feature_upsample[i](decoder)
            x_de = self.decoder_feature_conv[i](x_de)
            x_de_features.insert(0, x_de)

        if len(x_de_features) == 0:
            x_de_features = None

        # Sum list of tensors
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
                  conv_layer_multiplayer: int,
                  components: str,
                  no_groups: int,
                  dropout: Optional[float] = None,
                  sizes=None,
                  deconv_module='CNN'):
    """
    Create decoder block from number of convolution and convolution modules.
    Decoder block is followed by upsampling(interpolation) and joining with
    torch.cat()

    Args:
        conv_layers: Number of deconvolution layers
        conv_layer_multiplayer: number of convolution block in each layer.
        conv_kernel: Kernel size for deconvolution
        padding: Padding size for deconvolution
        components: String of components for building deconvolution
        no_groups: Number of groups for nn.GroupNorm
        up_sampling: If True the up-sampling is applied
        deconv_module: Module of the deconvolution for decoder

    Returns: nn.ModuleList
    """
    decoders = []
    feature_map = number_of_features_per_level(conv_layer_multiplayer,
                                               conv_layers)
    feature_map = list(reversed(feature_map))

    if deconv_module == 'CNN':
        for i in range(len(feature_map) - 1):
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            decoder = DecoderBlockCNN(in_ch=in_ch,
                                      out_ch=out_ch,
                                      size=size,
                                      dropout=dropout,
                                      components=components,
                                      no_group=no_groups)
            decoders.append(decoder)

    elif deconv_module == 'RCNN':
        for i in range(len(feature_map) - 1):
            in_ch = feature_map[i]
            out_ch = feature_map[i + 1]
            size = sizes[i]

            decoder = DecoderBlockRCNN(in_ch=in_ch,
                                       out_ch=out_ch,
                                       size=size,
                                       dropout=dropout,
                                       components=components,
                                       no_group=no_groups)

            decoders.append(decoder)

    elif deconv_module == 'unet3plus':
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
                                            no_group=no_groups,
                                            no_layer=conv_layers,
                                            encoder_feature_ch=encoder_feature_ch,
                                            decoder_feature_ch=decoder_feature_ch,
                                            dropout=dropout)
            decoders.append(decoder)

    return nn.ModuleList(decoders)
