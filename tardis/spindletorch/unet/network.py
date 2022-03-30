from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tardis.spindletorch.unet.convolution import (DoubleConvolution,
                                                  RecurrentDoubleConvolution)
from tardis.spindletorch.unet.decoder_blocks import build_decoder
from tardis.spindletorch.unet.encoder_blocks import build_encoder
from tardis.spindletorch.utils.utils import number_of_features_per_level


class UNet(nn.Module):
    """
    3D UNet model applied from:
    "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    <https://arxiv.org/pdf/1606.06650.pdf>.

    Args:
        in_channels: Number of input channels for first convolution
            default: 1
        out_channels: Number of output channels for last deconvolution
            default: 2 <- binary 0-1 segmentation
        sigmoid: If True, use nn.Sigmoid or nn.Softmax if False. Use True if
        nn.BCELoss is used as loss function for (two-class segmentation).
            default: True
        conv_module: Convolution module used for build network
        no_conv_layer: Number of convolution and deconvolution steps. Number of
        input channels for convolution is calculated as a linear progression.
        E.g. [64, 128, 256, 512]
            default: 4
        no_groups: Number of group for nn.GroupNorm
            default: 8
        conv_kernel: Kernel size for the convolution
            default: 3
        padding: Padding size for convolution
            default: 1
    Returns if Training:
        5D torch without final activation
    Returns if Prediction:
        If sigmoid is True
            5D torch with final activation from nn.Sigmoid()
        If sigmoid is False
            5D torch with final activation from nn.Softmax(dim=1)

    author: Robert Kiewisz
        modified of <https://github.com/wolny/pytorch-3dunet>
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 sigmoid=True,
                 no_conv_layer=5,
                 conv_layer_multiplayer=64,
                 patch_size=64,
                 layer_components="gcl",
                 dropout=None,
                 no_groups=8,
                 prediction=False):
        super(UNet, self).__init__()

        patch_sizes = [patch_size]
        for i in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        self.prediction = prediction
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=3,
                                     padding=1,
                                     dropout=dropout,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=2,
                                     conv_module=DoubleConvolution)

        self.decoder = build_decoder(conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     components=layer_components,
                                     sizes=patch_sizes,
                                     no_groups=no_groups,
                                     deconv_module='CNN')

        self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                          out_channels=out_channels,
                                          kernel_size=(1, 1, 1))

        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        encoder_features = []

        """ Encoder """
        for encoder in self.encoder:
            x = encoder(x)
            encoder_features.insert(0, x)

        encoder_features = encoder_features[1:]

        """ Decoder """
        for decoder, encoder_features in zip(self.decoder, encoder_features):
            x = decoder(encoder_features, x)

        x = self.final_conv_layer(x)

        if self.prediction:
            x = self.activation(x)

        return x


class ResUNet(nn.Module):
    """
    3D ResUNET MODEL

    author: Robert Kiewisz
        modified of <10.1016/j.isprsjprs.2020.01.013>
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 sigmoid=True,
                 patch_size=64,
                 no_conv_layer=4,
                 conv_layer_multiplayer=64,
                 layer_components="cgl",
                 dropout: Optional[float] = None,
                 no_groups=8,
                 prediction=False):
        super(ResUNet, self).__init__()

        patch_sizes = [patch_size]
        for i in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        self.prediction = prediction
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=3,
                                     padding=1,
                                     dropout=dropout,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=2,
                                     conv_module=RecurrentDoubleConvolution)

        self.decoder = build_decoder(conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     components=layer_components,
                                     sizes=patch_sizes,
                                     dropout=dropout,
                                     no_groups=no_groups,
                                     deconv_module='RCNN')

        self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                          out_channels=out_channels,
                                          kernel_size=(1, 1, 1))

        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        encoder_features = []

        """ Encoder """
        for encoder in self.encoder:
            x = encoder(x)
            encoder_features.insert(0, x)

        encoder_features = encoder_features[1:]

        """ Decoder """
        for decoder, encoder_features in zip(self.decoder, encoder_features):
            x = decoder(encoder_features, x)

        x = self.final_conv_layer(x)

        if self.prediction:
            x = self.activation(x)

        return x


class UNet3Plus(nn.Module):
    """
    3D Fully Connected UNet

    author: Robert Kiewisz
        modified of <https://arxiv.org/abs/2004.08790>
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 sigmoid=True,
                 patch_size=64,
                 no_conv_layer=4,
                 conv_layer_multiplayer=64,
                 layer_components="cgl",
                 no_groups=8,
                 conv_kernel=3,
                 pool_kernel=2,
                 padding=1,
                 classifies=False,
                 prediction=False):
        super(UNet3Plus, self).__init__()

        patch_sizes = [patch_size]
        for i in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        feature_map = number_of_features_per_level(conv_layer_multiplayer,
                                                   no_conv_layer)

        self.prediction = prediction
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=conv_kernel,
                                     padding=padding,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=pool_kernel,
                                     conv_module=DoubleConvolution)

        if classifies:
            self.cls = nn.Sequential(nn.Dropout(p=0.5),
                                     nn.Conv3d(in_channels=feature_map[len(feature_map) - 1],
                                               out_channels=2,
                                               kernel_size=1),
                                     nn.AdaptiveAvgPool3d(output_size=1),
                                     nn.Sigmoid())
        else:
            self.cls = None

        self.decoder = build_decoder(conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     components=layer_components,
                                     sizes=patch_sizes,
                                     no_groups=no_groups,
                                     deconv_module='unet3plus')

        self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                          out_channels=out_channels,
                                          kernel_size=(1, 1, 1))

        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    @staticmethod
    def dotProduct(x, x_cls):
        B, N, D, H, W = x.shape
        x = x.view(B, N, D * H * W)

        final = torch.einsum("ijk,ijk->ijk", [x, x_cls])
        final = final.view(B, N, D, H, W)

        return final

    def forward(self,
                x: torch.Tensor):
        """ Encoder """
        encoder_features = []

        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            # Don't save last encoder layer
            # Encoder layers are inverted
            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Classification on last encoder """
        if self.cls is not None:
            x_cls = self.cls(x).squeeze(3).squeeze(2)
            x_cls_max = x_cls.argmax(dim=1)
            x_cls_max = x_cls_max[:, np.newaxis].float()

        """ Decoder """
        for decoder in self.decoder:
            decoder_features = [x]

            x = decoder(x=x,
                        encoder_features=encoder_features,
                        decoder_features=decoder_features[2:])

            # add/remove layer at each iter
            decoder_features.insert(0, x)
            encoder_features = encoder_features[1:]

        # for i, decoder in enumerate(decoder_features):
        if self.cls is not None:
            x = self.final_conv_layer(x)
            x = self.dotProduct(x, x_cls_max)

            if self.prediction:
                x = self.activation(x)

            return x, x_cls
        else:
            x = self.final_conv_layer(x)

            if self.prediction:
                x = self.activation(x)

            return x
