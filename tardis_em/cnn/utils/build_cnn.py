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
import torch
import torch.nn as nn

from tardis_em.cnn.model.convolution import (
    DoubleConvolution,
    RecurrentDoubleConvolution,
)

from tardis_em.cnn.model.decoder_blocks import build_decoder
from tardis_em.cnn.model.encoder_blocks import build_encoder
from tardis_em.cnn.utils.utils import number_of_features_per_level


class BasicCNN(nn.Module):
    """
    Basic CNN MODEL

    Args:
        in_channels (int): Number of input channels for the first convolution.
        out_channels (int): Number of output channels for the last deconvolution.
        sigmoid (bool): If True, use nn.Sigmoid or nn.Softmax if False. Use True if
            nn.BCELoss is used as a loss function for (two-class segmentation).
        num_conv_layer (int): Number of convolution and deconvolution steps. A number of
            input channels for convolution is calculated as a linear progression.
            E.g. [64, 128, 256, 512].
        conv_layer_scaler (int): Scaler for the output feature channels.
        conv_kernel (int): Kernel size for the convolution.
        padding (int): Padding size for convolution.
        pool_kernel (int): kernel size for max_pooling.
        img_patch_size (int): Image patch size used for calculation network structure.
        layer_components (str): Convolution module used for building network.
        dropout (float, optional): If float, the dropout layer is built with a given drop out rate.
        num_group (int): Number of groups for nn.GroupNorm.
        prediction (bool): If True, prediction mode is on.

    Returns if Training:
        torch.tensor: 5D torch without final activation.

    Returns if Prediction:
        If sigmoid is True:
            torch.tensor: 5D torch with final activation from nn.Sigmoid().
        If sigmoid is False:
            torch.tensor: 5D torch with final activation from nn.Softmax(dim=1).
    """

    def __init__(
        self,
        model="CNN",
        in_channels=1,
        out_channels=1,
        sigmoid=True,
        num_conv_layer=5,
        conv_layer_scaler=64,
        conv_kernel=3,
        padding=1,
        pool_kernel=2,
        img_patch_size=64,
        layer_components="3gcl",
        dropout=None,
        num_group=8,
        prediction=False,
    ):
        super(BasicCNN, self).__init__()
        self.prediction = prediction
        self.model = model

        patch_sizes = [img_patch_size]
        for _ in range(num_conv_layer):
            img_patch_size = int(img_patch_size / 2)
            patch_sizes.append(img_patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder """
        if self.model == "CNN":
            self.encoder = build_encoder(
                in_ch=in_channels,
                conv_layers=num_conv_layer,
                conv_layer_scaler=conv_layer_scaler,
                conv_kernel=conv_kernel,
                padding=padding,
                dropout=dropout,
                num_group=num_group,
                components=layer_components,
                pool_kernel=pool_kernel,
                conv_module=DoubleConvolution,
            )
        elif self.model == "RCNN":
            self.encoder = build_encoder(
                in_ch=in_channels,
                conv_layers=num_conv_layer,
                conv_layer_scaler=conv_layer_scaler,
                conv_kernel=conv_kernel,
                padding=padding,
                dropout=dropout,
                num_group=num_group,
                components=layer_components,
                pool_kernel=pool_kernel,
                conv_module=RecurrentDoubleConvolution,
            )
        """ Decoder """
        self.decoder = build_decoder(
            conv_layers=num_conv_layer,
            conv_layer_scaler=conv_layer_scaler,
            components=layer_components,
            conv_kernel=conv_kernel,
            padding=padding,
            sizes=patch_sizes,
            num_group=num_group,
            deconv_module=model,
        )

        """ Final Layer """
        if "3" in layer_components:
            self.final_conv_layer = nn.Conv3d(
                in_channels=conv_layer_scaler, out_channels=out_channels, kernel_size=1
            )
        elif "2" in layer_components:
            self.final_conv_layer = nn.Conv2d(
                in_channels=conv_layer_scaler, out_channels=out_channels, kernel_size=1
            )

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)


class UNet(BasicCNN):
    """
    2D/3D UNET MODEL

    "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    <https://arxiv.org/pdf/1606.06650.pdf>.
    """

    def __init__(self, model="CNN", **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward for Unet model.

            Args:
                x (torch.Tensor): Input image features.

            Returns:
                torch.Tensor: Probability mask of predicted image.
        """
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
        x = self.final_conv_layer(x)

        if self.prediction:
            return self.activation(x)
        else:
            return x


class ResUNet(BasicCNN):
    """
    2D/3D RESNET MODEL

    modified of <10.1016/j.isprsjprs.2020.01.013>
    """

    def __init__(self, model="RCNN", **kwargs):
        super(ResUNet, self).__init__(**kwargs)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward for ResNet model.

            Args:
                x (torch.Tensor): Input image features.

            Returns:
                torch.Tensor: Probability mask of predicted image.
        """
        encoder_features = []

        """ Encoder """
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Decoder """
        for decoder, encoder_features in zip(self.decoder, encoder_features):
            x = decoder(encoder_features, x)

        """ Final layer and Prediction """
        x = self.final_conv_layer(x)

        if self.prediction:
            return self.activation(x)
        else:
            return x


class UNet3Plus(nn.Module):
    """
    3D FULLY CONNECTED UNET

    modified of <https://arxiv.org/abs/2004.08790>
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        sigmoid=True,
        num_conv_layer=5,
        conv_layer_scaler=64,
        conv_kernel=3,
        padding=1,
        pool_kernel=2,
        img_patch_size=64,
        layer_components="3gcl",
        dropout=None,
        num_group=8,
        prediction=False,
        classifies=False,
        decoder_features=None,
    ):
        super(UNet3Plus, self).__init__()
        self.prediction = prediction
        self.decoder_features = decoder_features

        patch_sizes = [img_patch_size]
        for _ in range(num_conv_layer):
            img_patch_size = int(img_patch_size / 2)
            patch_sizes.append(img_patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        feature_map = number_of_features_per_level(conv_layer_scaler, num_conv_layer)

        """ Encoder """
        self.encoder = build_encoder(
            in_ch=in_channels,
            conv_layers=num_conv_layer,
            conv_layer_scaler=conv_layer_scaler,
            conv_kernel=conv_kernel,
            padding=padding,
            num_group=num_group,
            dropout=dropout,
            components=layer_components,
            pool_kernel=pool_kernel,
            conv_module=DoubleConvolution,
        )

        """ UNet3Plus classifier """
        if classifies:
            if "3" in layer_components:
                self.cls = nn.Sequential(
                    nn.Dropout(),
                    nn.Conv3d(
                        in_channels=feature_map[len(feature_map) - 1],
                        out_channels=2,
                        kernel_size=1,
                    ),
                    nn.AdaptiveAvgPool3d(output_size=1),
                    nn.Sigmoid(),
                )
            elif "2" in layer_components:
                self.cls = nn.Sequential(
                    nn.Dropout(),
                    nn.Conv2d(
                        in_channels=feature_map[len(feature_map) - 1],
                        out_channels=2,
                        kernel_size=1,
                    ),
                    nn.AdaptiveAvgPool2d(output_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.cls = None

        """ Decoder """
        self.decoder = build_decoder(
            conv_layers=num_conv_layer,
            conv_layer_scaler=conv_layer_scaler,
            components=layer_components,
            conv_kernel=conv_kernel,
            padding=padding,
            sizes=patch_sizes,
            num_group=num_group,
            deconv_module="unet3plus",
        )

        """ Final Layer """
        if "3" in layer_components:
            self.final_conv_layer = nn.Conv3d(
                in_channels=conv_layer_scaler, out_channels=out_channels, kernel_size=1
            )
        elif "2" in layer_components:
            self.final_conv_layer = nn.Conv2d(
                in_channels=conv_layer_scaler, out_channels=out_channels, kernel_size=1
            )

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    @staticmethod
    def dot_product(x: torch.Tensor, x_cls: torch.Tensor) -> torch.Tensor:
        """
        Dot product for two tensors.

        Args:
            x (torch.Tensor): Image tensor.
            x_cls (torch.Tensor): Classified image tensor.

        Returns:
            torch.Tensor: Dot product of two tensors.
        """
        B, N, D, H, W = x.shape
        x = x.view(B, N, D * H * W)

        final = torch.einsum("ijk,ijk->ijk", [x, x_cls])
        final = final.view(B, N, D, H, W)

        return final

    def forward(self, x: torch.Tensor):
        """
        Forward for Unet3Plus model.

            Args:
                x (torch.Tensor): Input image features.

            Returns:
                torch.Tensor: Probability mask of predicted image.
        """
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
        decoder_features = []
        for decoder in self.decoder:
            x = decoder(
                x=x,
                encoder_features=encoder_features,
                decoder_features=decoder_features,
            )

            # add/remove layer at each iter
            decoder_features.insert(0, x)
            encoder_features = encoder_features[1:]

        """ Final Layer / Prediction"""
        if self.cls is not None:
            x = self.dot_product(self.final_conv_layer(x), x_cls_max)

            if self.prediction:
                x = self.activation(x)

            return x, x_cls
        else:
            x = self.final_conv_layer(x)

            if self.prediction:
                return self.activation(x)
            else:
                return x


class FNet(nn.Module):
    """
    New Unet model combining Unet and Unet3Plus
    The model shares encoder path which is split for decoding patch Unet and Unet3Plus
    style. The final layers from each are summed and sigmoid

    Args:
        in_channels: Number of input channels for the first convolution.
        out_channels: Number of output channels for the last deconvolution.
        sigmoid: If True, use nn.Sigmoid or nn.Softmax if False. Use True if
            nn.BCELoss is used as a loss function for (two-class segmentation).
        num_conv_layer: Number of convolution and deconvolution steps. A number of
            input channels for convolution is calculated as a linear progression.
            E.g. [64, 128, 256, 512].
        conv_layer_scaler: Feature output of the first layer.
        conv_kernel: Kernel size for the convolution.
        padding: Padding size for convolution.
        pool_kernel: kernel size for max_pooling.
        img_patch_size: Image patch size used for calculation of network structure.
        layer_components: Convolution module used for building network.
        num_group: Number of groups for nn.GroupNorm.
        prediction: If True, prediction mode is on.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        sigmoid=True,
        num_conv_layer=5,
        conv_layer_scaler=64,
        conv_kernel=3,
        padding=1,
        pool_kernel=2,
        dropout=None,
        img_patch_size=64,
        layer_components="3gcl",
        num_group=8,
        attn_features=False,
        prediction=False,
    ):
        super(FNet, self).__init__()
        self.prediction = prediction
        self.attn_features = attn_features

        patch_sizes = [img_patch_size]
        for _ in range(num_conv_layer):
            img_patch_size = int(img_patch_size / 2)
            patch_sizes.append(img_patch_size)
        patch_sizes = list(reversed(patch_sizes))[2:]

        """ Encoder """
        self.encoder = build_encoder(
            in_ch=in_channels,
            conv_layers=num_conv_layer,
            conv_layer_scaler=conv_layer_scaler,
            conv_kernel=conv_kernel,
            padding=padding,
            dropout=dropout,
            num_group=num_group,
            components=layer_components,
            pool_kernel=pool_kernel,
            conv_module=DoubleConvolution,
            attn_features=self.attn_features,
        )

        """ Decoder """
        self.decoder_unet = build_decoder(
            conv_layers=num_conv_layer,
            conv_layer_scaler=conv_layer_scaler,
            components=layer_components,
            conv_kernel=conv_kernel,
            padding=padding,
            sizes=patch_sizes,
            num_group=num_group,
        )
        self.decoder_3plus = build_decoder(
            conv_layers=num_conv_layer,
            conv_layer_scaler=conv_layer_scaler,
            components=layer_components,
            conv_kernel=conv_kernel,
            padding=padding,
            sizes=patch_sizes,
            num_group=num_group,
            deconv_module="unet3plus",
            attn_features=self.attn_features,
        )

        """ Final Layer """
        if "3" in layer_components:
            self.unet_conv_layer = nn.Conv3d(
                in_channels=conv_layer_scaler,
                out_channels=conv_layer_scaler,
                kernel_size=1,
            )
            self.unet3plus_conv_layer = nn.Conv3d(
                in_channels=conv_layer_scaler,
                out_channels=conv_layer_scaler,
                kernel_size=1,
            )

            self.final_conv_layer = nn.Conv3d(
                in_channels=conv_layer_scaler * 2,
                out_channels=out_channels,
                kernel_size=1,
            )
        elif "2" in layer_components:
            self.unet_conv_layer = nn.Conv2d(
                in_channels=conv_layer_scaler,
                out_channels=conv_layer_scaler,
                kernel_size=1,
            )
            self.unet3plus_conv_layer = nn.Conv2d(
                in_channels=conv_layer_scaler,
                out_channels=conv_layer_scaler,
                kernel_size=1,
            )

            self.final_conv_layer = nn.Conv2d(
                in_channels=conv_layer_scaler * 2,
                out_channels=out_channels,
                kernel_size=1,
            )

        """ Prediction """
        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """
        Forward for FNet model.

            Args:
                x (torch.Tensor): Input image features.

            Returns:
                torch.Tensor: Probability mask of predicted image.
        """
        encoder_features = []

        """ Encoder """
        for i, encoder in enumerate(self.encoder):
            x = encoder(x)

            if (len(self.encoder) - 1) != i:
                encoder_features.insert(0, x)

        """ Decoders """
        x_3plus = x
        for i, (decoder, decoder_2) in enumerate(
            zip(self.decoder_unet, self.decoder_3plus)
        ):
            # Add Decoder layer
            x = decoder(encoder_features[0], x)
            x_3plus = decoder_2(
                x=x_3plus,
                encoder_features=encoder_features,
            )

            # Remove layer at each iter
            encoder_features = encoder_features[1:]

        """ Final Layer/Prediction """
        x = self.final_conv_layer(
            torch.cat(
                (self.unet_conv_layer(x), self.unet3plus_conv_layer(x_3plus)), dim=1
            )
        )

        if self.prediction:
            return self.activation(x)
        else:
            return x
