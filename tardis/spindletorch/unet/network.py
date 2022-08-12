from base64 import decode
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
        out_channels: Number of output channels for last deconvolution
        sigmoid: If True, use nn.Sigmoid or nn.Softmax if False. Use True if
            nn.BCELoss is used as loss function for (two-class segmentation)
        no_conv_layer: Number of convolution and deconvolution steps. Number of
            input channels for convolution is calculated as a linear progression.
            E.g. [64, 128, 256, 512]
        conv_layer_multiplayer: Feature output of first layer
        conv_kernel: Kernel size for the convolution
        padding: Padding size for convolution
        maxpool_kernel: kernel size for max_pooling
        patch_size: Image patch size used for calculation network structure
        layer_components: Convolution module used for build network
        dropout: If float, dropout layer is build with given drop out rate
        no_groups: Number of group for nn.GroupNorm
        prediction: If True, prediction mode is on

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
                 conv_kernel=3,
                 padding=1,
                 pool_kernel=2,
                 patch_size=64,
                 layer_components="3gcl",
                 dropout=None,
                 no_groups=8,
                 prediction=False):
        super(UNet, self).__init__()
        self.prediction = prediction

        patch_sizes = [patch_size]
        for _ in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder """
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=conv_kernel,
                                     padding=padding,
                                     dropout=dropout,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=pool_kernel,
                                     conv_module=DoubleConvolution)

        """ Decoder """
        self.decoder = build_decoder(conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     components=layer_components,
                                     sizes=patch_sizes,
                                     no_groups=no_groups,
                                     deconv_module='CNN')

        """ Final Layer """
        if '3' in layer_components:
            self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)
        elif '2' in layer_components:
            self.final_conv_layer = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        encoder_features = []

        """ Encoder """
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Decoder """
        for decoder, encoder_features in zip(self.decoder, encoder_features):
            x = decoder(encoder_features, x)

        """ Prediction """
        if self.prediction:
            return self.activation(self.final_conv_layer(x))
        else:
            return self.final_conv_layer(x)


class WNet(nn.Module):
    """
    WNet 

    author: Robert Kiewisz
        modified of <10.1016/j.isprsjprs.2020.01.013>
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 sigmoid=True,
                 no_conv_layer=5,
                 conv_layer_multiplayer=64,
                 conv_kernel=3,
                 padding=1,
                 pool_kernel=2,
                 patch_size=64,
                 layer_components="3gcl",
                 dropout=None,
                 no_groups=8,
                 prediction=False):
        super(WNet, self).__init__()
        self.prediction = prediction

        patch_sizes = [patch_size]
        for _ in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder - Encoder """
        self.E_encoder = build_encoder(in_ch=in_channels,
                                       conv_layers=no_conv_layer,
                                       conv_layer_multiplayer=conv_layer_multiplayer,
                                       conv_kernel=conv_kernel,
                                       padding=padding,
                                       dropout=dropout,
                                       no_groups=no_groups,
                                       components=layer_components,
                                       pool_kernel=pool_kernel,
                                       conv_module=DoubleConvolution)

        """ Encoder - Decoder """
        self.E_decoder = build_decoder(conv_layers=no_conv_layer,
                                       conv_layer_multiplayer=conv_layer_multiplayer,
                                       components=layer_components,
                                       sizes=patch_sizes,
                                       no_groups=no_groups,
                                       deconv_module='CNN')

        """ Encoder - Final Layer """
        if '3' in layer_components:
            self.E_final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                                out_channels=out_channels,
                                                kernel_size=1)
        elif '2' in layer_components:
            self.E_final_conv_layer = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                                out_channels=out_channels,
                                                kernel_size=1)
        self.E_activation = nn.Softmax(dim=1)

        """ Decoder - Encoder """
        self.D_encoder = build_encoder(in_ch=in_channels,
                                       conv_layers=no_conv_layer,
                                       conv_layer_multiplayer=conv_layer_multiplayer,
                                       conv_kernel=conv_kernel,
                                       padding=padding,
                                       dropout=dropout,
                                       no_groups=no_groups,
                                       components=layer_components,
                                       pool_kernel=pool_kernel,
                                       conv_module=DoubleConvolution)

        """ Decoder - Decoder """
        self.D_decoder = build_decoder(conv_layers=no_conv_layer,
                                       conv_layer_multiplayer=conv_layer_multiplayer,
                                       components=layer_components,
                                       sizes=patch_sizes,
                                       no_groups=no_groups,
                                       deconv_module='CNN')

        """ Decoder - Final Layer """
        if '3' in layer_components:
            self.D_final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                                out_channels=out_channels,
                                                kernel_size=1)
        elif '2' in layer_components:
            self.D_final_conv_layer = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                                out_channels=out_channels,
                                                kernel_size=1)

        """ Prediction """
        if sigmoid:
            self.D_activation = nn.Sigmoid()
        else:
            self.D_activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        """ WUnet - Encoder """
        encoder_features = []

        # Encoder
        for i, encoder in enumerate(self.E_encoder):
            x = encoder(x)

            if (len(self.E_encoder) - 1) != i:
                encoder_features.insert(0, x)

        # Decoder
        for decoder, encoder_features in zip(self.E_decoder, encoder_features):
            x = decoder(encoder_features, x)

        x = self.E_activation(self.E_final_conv_layer(x))

        """ WUnet - Decoder """
        encoder_features = []

        # Encoder
        for i, encoder in enumerate(self.D_encoder):
            x = encoder(x)

            if (len(self.D_encoder) - 1) != i:
                encoder_features.insert(0, x)

        # Decoder
        for decoder, encoder_features in zip(self.D_decoder, encoder_features):
            x = decoder(encoder_features, x)

        """ Prediction """
        if self.prediction:
            return self.D_activation(self.D_final_conv_layer(x))
        else:
            return self.D_final_conv_layer(x)


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
                 conv_kernel=3,
                 padding=1,
                 pool_kernel=2,
                 conv_layer_multiplayer=64,
                 layer_components="3cgl",
                 dropout: Optional[float] = None,
                 no_groups=8,
                 prediction=False):
        super(ResUNet, self).__init__()
        self.prediction = prediction

        patch_sizes = [patch_size]
        for _ in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)

        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder """
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=conv_kernel,
                                     padding=padding,
                                     dropout=dropout,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=pool_kernel,
                                     conv_module=RecurrentDoubleConvolution)

        self.decoder = build_decoder(conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     components=layer_components,
                                     sizes=patch_sizes,
                                     dropout=dropout,
                                     no_groups=no_groups,
                                     deconv_module='RCNN')

        """ Final Layer """
        if '3' in layer_components:
            self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)
        elif '2' in layer_components:
            self.final_conv_layer = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        encoder_features = []

        """ Encoder """
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Decoder """
        for decoder, encoder_features in zip(self.decoder, encoder_features):
            x = decoder(encoder_features, x)

        """ Final Layer """
        x = self.final_conv_layer(x)

        """ Prediction """
        if self.prediction:
            return self.activation(self.final_conv_layer(x))
        else:
            return self.final_conv_layer(x)


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
                 layer_components="3cgl",
                 no_groups=8,
                 conv_kernel=3,
                 pool_kernel=2,
                 padding=1,
                 classifies=False,
                 prediction=False):
        super(UNet3Plus, self).__init__()
        self.prediction = prediction

        patch_sizes = [patch_size]
        for _ in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)

        patch_sizes = list(reversed(patch_sizes))[2:]

        feature_map = number_of_features_per_level(conv_layer_multiplayer,
                                                   no_conv_layer)

        """ Encoder """
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=conv_kernel,
                                     padding=padding,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=pool_kernel,
                                     conv_module=DoubleConvolution)

        """ UNet3Plus classifier """
        if classifies:
            if '3' in layer_components:
                self.cls = nn.Sequential(nn.Dropout(p=0.5),
                                         nn.Conv3d(in_channels=feature_map[len(feature_map) - 1],
                                                   out_channels=2,
                                                   kernel_size=1),
                                         nn.AdaptiveAvgPool3d(output_size=1),
                                         nn.Sigmoid())
            elif '2' in layer_components:
                self.cls = nn.Sequential(nn.Dropout(p=0.5),
                                         nn.Conv2d(in_channels=feature_map[len(feature_map) - 1],
                                                   out_channels=2,
                                                   kernel_size=1),
                                         nn.AdaptiveAvgPool2d(output_size=1),
                                         nn.Sigmoid())
        else:
            self.cls = None

        """ Decoder """
        self.decoder = build_decoder(conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     components=layer_components,
                                     sizes=patch_sizes,
                                     no_groups=no_groups,
                                     deconv_module='unet3plus')

        """ Final Layer """
        if '3' in layer_components:
            self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)
        elif '2' in layer_components:
            self.final_conv_layer = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)

        """ Prediction """
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

        """ Final Layer / Prediction"""
        if self.cls is not None:
            x = self.dotProduct(self.final_conv_layer(x), x_cls_max)

            if self.prediction:
                x = self.activation(x)

            return x, x_cls
        else:
            if self.prediction:
                return self.activation(self.final_conv_layer(x))
            else:
                return self.final_conv_layer(x)


class Big_UNet(nn.Module):
    """
    New Unet model combining Unet and Unet3Plus
    Model shares encoder path which is splitted for decoding patch Unet and Unet3Plus
    style. Final layers from each are summed and sigmoid

    Args:
        in_channels: Number of input channels for first convolution
        out_channels: Number of output channels for last deconvolution
        sigmoid: If True, use nn.Sigmoid or nn.Softmax if False. Use True if
            nn.BCELoss is used as loss function for (two-class segmentation)
        no_conv_layer: Number of convolution and deconvolution steps. Number of
            input channels for convolution is calculated as a linear progression.
            E.g. [64, 128, 256, 512]
        conv_layer_multiplayer: Feature output of first layer
        conv_kernel: Kernel size for the convolution
        padding: Padding size for convolution
        maxpool_kernel: kernel size for max_pooling
        patch_size: Image patch size used for calculation network structure
        layer_components: Convolution module used for build network
        no_groups: Number of group for nn.GroupNorm
        prediction: If True, prediction mode is on
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 sigmoid=True,
                 no_conv_layer=5,
                 conv_layer_multiplayer=64,
                 conv_kernel=3,
                 padding=1,
                 pool_kernel=2,
                 patch_size=64,
                 layer_components="3gcl",
                 no_groups=8,
                 prediction=False):
        super(Big_UNet, self).__init__()
        self.prediction = prediction

        patch_sizes = [patch_size]
        for _ in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder """
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=conv_kernel,
                                     padding=padding,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=pool_kernel,
                                     conv_module=DoubleConvolution)

        """ Decoder """
        self.decoder_unet = build_decoder(conv_layers=no_conv_layer,
                                          conv_layer_multiplayer=conv_layer_multiplayer,
                                          components=layer_components,
                                          sizes=patch_sizes,
                                          no_groups=no_groups,
                                          deconv_module='CNN')
        self.decoder_3plus = build_decoder(conv_layers=no_conv_layer,
                                           conv_layer_multiplayer=conv_layer_multiplayer,
                                           components=layer_components,
                                           sizes=patch_sizes,
                                           no_groups=no_groups,
                                           deconv_module='unet3plus')

        """ Final Layer """
        if '3' in layer_components:
            self.final_conv_layer_unet = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                                   out_channels=out_channels,
                                                   kernel_size=1)
            self.final_conv_layer_3plus = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                                    out_channels=out_channels,
                                                    kernel_size=1)
        elif '2' in layer_components:
            self.final_conv_layer_unet = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                                   out_channels=out_channels,
                                                   kernel_size=1)
            self.final_conv_layer_3plus = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                                    out_channels=out_channels,
                                                    kernel_size=1)

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        encoder_features = []

        """ Encoder """
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Decoder UNet """
        x_unet = x
        for decoder, features in zip(self.decoder_unet, encoder_features):
            x_unet = decoder(features, x_unet)

        """ Decoder UNet_3Plus """
        x_3plus = x
        for decoder in self.decoder_3plus:
            decoder_features = [x_3plus]

            x_3plus = decoder(x=x_3plus,
                              encoder_features=encoder_features,
                              decoder_features=decoder_features[2:])

            # add/remove layer at each iter
            decoder_features.insert(0, x_3plus)
            encoder_features = encoder_features[1:]

        """ Final Layer/Prediction """
        if self.prediction:
            return self.activation(self.final_conv_layer_unet(x_unet) + self.final_conv_layer_3plus(x_3plus))
        else:
            return self.final_conv_layer_unet(x_unet) + self.final_conv_layer_3plus(x_3plus)


class FNet(nn.Module):
    """
    New Unet model combining Unet and Unet3Plus
    Model shares encoder path which is splitted for decoding patch Unet and Unet3Plus
    style. Final layers from each are summed and sigmoid

    Args:
        in_channels: Number of input channels for first convolution
        out_channels: Number of output channels for last deconvolution
        sigmoid: If True, use nn.Sigmoid or nn.Softmax if False. Use True if
            nn.BCELoss is used as loss function for (two-class segmentation)
        no_conv_layer: Number of convolution and deconvolution steps. Number of
            input channels for convolution is calculated as a linear progression.
            E.g. [64, 128, 256, 512]
        conv_layer_multiplayer: Feature output of first layer
        conv_kernel: Kernel size for the convolution
        padding: Padding size for convolution
        maxpool_kernel: kernel size for max_pooling
        patch_size: Image patch size used for calculation network structure
        layer_components: Convolution module used for build network
        no_groups: Number of group for nn.GroupNorm
        prediction: If True, prediction mode is on
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 sigmoid=True,
                 no_conv_layer=5,
                 conv_layer_multiplayer=64,
                 conv_kernel=3,
                 padding=1,
                 pool_kernel=2,
                 patch_size=64,
                 layer_components="3gcl",
                 no_groups=8,
                 prediction=False):
        super(FNet, self).__init__()
        self.prediction = prediction

        patch_sizes = [patch_size]
        for _ in range(no_conv_layer):
            patch_size = int(patch_size / 2)
            patch_sizes.append(patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder """
        self.encoder = build_encoder(in_ch=in_channels,
                                     conv_layers=no_conv_layer,
                                     conv_layer_multiplayer=conv_layer_multiplayer,
                                     conv_kernel=conv_kernel,
                                     padding=padding,
                                     no_groups=no_groups,
                                     components=layer_components,
                                     pool_kernel=pool_kernel,
                                     conv_module=DoubleConvolution)

        """ Decoder """
        self.decoder_unet = build_decoder(conv_layers=no_conv_layer,
                                          conv_layer_multiplayer=conv_layer_multiplayer,
                                          components=layer_components,
                                          sizes=patch_sizes,
                                          no_groups=no_groups,
                                          deconv_module='CNN')
        self.decoder_3plus = build_decoder(conv_layers=no_conv_layer,
                                           conv_layer_multiplayer=conv_layer_multiplayer,
                                           components=layer_components,
                                           sizes=patch_sizes,
                                           no_groups=no_groups,
                                           deconv_module='unet3plus')

        """ Final Layer """
        if '3' in layer_components:
            self.final_conv_layer = nn.Conv3d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)
        elif '2' in layer_components:
            self.final_conv_layer = nn.Conv2d(in_channels=conv_layer_multiplayer,
                                              out_channels=out_channels,
                                              kernel_size=1)

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self,
                x: torch.Tensor):
        encoder_features = []

        """ Encoder """
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Decoder UNet """
        x_unet = x
        for decoder, features in zip(self.decoder_unet, encoder_features):
            x_unet = decoder(features, x_unet)

        """ Decoder UNet_3Plus """
        x_3plus = x
        for decoder in self.decoder_3plus:
            decoder_features = [x_3plus]

            x_3plus = decoder(x=x_3plus,
                              encoder_features=encoder_features,
                              decoder_features=decoder_features[2:])

            # add/remove layer at each iter
            decoder_features.insert(0, x_3plus)
            encoder_features = encoder_features[1:]

        """ Final Layer/Prediction """
        if self.prediction:
            return self.activation(self.final_conv_layer(x_unet + x_3plus))
        else:
            return self.final_conv_layer(x_unet + x_3plus)
