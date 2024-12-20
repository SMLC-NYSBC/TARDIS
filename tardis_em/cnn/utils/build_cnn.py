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
    A neural network model class for constructing and managing a customizable Convolutional Neural Network (CNN)
    and its variations.

    This class serves as a flexible framework for building encoder and decoder pipelines, defining the architecture
    of convolutional layers, and configuring associated parameters. It allows for adaptable hyperparameter settings
    to tailor the neural network model to specific tasks, including image patch processing and segmentation tasks.
    Additionally, it manages activation functions and final layer outputs suitable for either classification or
    prediction purposes.
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
        """
        Initializes an instance of the BasicCNN class used for constructing and managing
        a Convolutional Neural Network (CNN) with configurable parameters such as the
        number of layers, kernel size, and pooling operations. This class is highly
        flexible and allows the user to define architectural details including the
        overall number of layers, groupings, and activation methods.

        :param model: Specifies the model type, default is "CNN".
        :type model: str
        :param in_channels: The number of input channels to the CNN.
        :type in_channels: int
        :param out_channels: The number of output channels produced by the CNN.
        :type out_channels: int
        :param sigmoid: Determines the use of a sigmoid activation function.
        :type sigmoid: bool
        :param num_conv_layer: The total number of convolutional layers in the model.
        :type num_conv_layer: int
        :param conv_layer_scaler: Scaling factor for the convolution layers.
        :type conv_layer_scaler: int
        :param conv_kernel: Size of the kernel used in the convolutional layers.
        :type conv_kernel: int
        :param padding: Padding applied to convolutional operations.
        :type padding: int
        :param pool_kernel: Kernel size used for pooling operations.
        :type pool_kernel: int
        :param img_patch_size: Size of the image patches processed by the CNN.
        :type img_patch_size: int
        :param layer_components: Describes the structure of components for each layer, e.g., "3gcl".
        :type layer_components: str
        :param dropout: Probability for dropout regularization if enabled.
        :type dropout: float or None
        :param num_group: Number of groups used for grouped convolution operations.
        :type num_group: int
        :param prediction: Indicates whether the model is being used for predictions.
        :type prediction: bool
        """
        super(BasicCNN, self).__init__()

        self.prediction = prediction
        self.model = model
        self.encoder, self.decoder = None, None
        self.patch_sizes = img_patch_size

        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_conv_layer = num_conv_layer
        self.conv_layer_scaler = conv_layer_scaler
        self.conv_kernel = conv_kernel
        self.padding = padding
        self.dropout = dropout
        self.num_group = num_group
        self.layer_components = layer_components
        self.pool_kernel = pool_kernel

        self.final_conv_layer, self.sigmoid, self.activation = None, sigmoid, None

        self.update_patch_size(img_patch_size, prediction)
        self.build_cnn_model()

    def update_patch_size(self, img_patch_size, sigmoid):
        """
        Updates the image patch sizes and rebuilds the model if necessary.

        This function recalculates the patch sizes for an image as it traverses through convolutional layers.
        It also rebuilds the CNN model if the current model is not of type string, retaining the previous
        model's state dictionary.

        :param img_patch_size: The initial size of the image patches to be processed.
        :type img_patch_size: int
        :param sigmoid: The sigmoid activation function applied to predictions to constrain outputs.
        :type sigmoid: Callable or function
        :return: None
        """
        self.prediction = sigmoid

        self.patch_sizes = [img_patch_size]
        for _ in range(self.num_conv_layer):
            img_patch_size = int(img_patch_size / 2)
            self.patch_sizes.append(img_patch_size)
        self.patch_sizes = list(reversed(self.patch_sizes))[2:]

        if not isinstance(self.model, str):
            states = self.model.state_dict()
            self.build_cnn_model()
            self.model.load_state_dict(states)

            states = None

    def build_cnn_model(self):
        """
        Constructs a Convolutional Neural Network (CNN) or Recurrent Convolutional Neural
        Network (RCNN) model by building encoder, decoder, and a final layer based on the
        configuration provided. This function handles the creation of the model
        architecture including layers, activation functions, and other components.
        """
        if self.model == "CNN":
            self.encoder = build_encoder(
                in_ch=self.in_channels,
                conv_layers=self.num_conv_layer,
                conv_layer_scaler=self.conv_layer_scaler,
                conv_kernel=self.conv_kernel,
                padding=self.padding,
                dropout=self.dropout,
                num_group=self.num_group,
                components=self.layer_components,
                pool_kernel=self.pool_kernel,
                conv_module=DoubleConvolution,
            )
        elif self.model == "RCNN":
            self.encoder = build_encoder(
                in_ch=self.in_channels,
                conv_layers=self.num_conv_layer,
                conv_layer_scaler=self.conv_layer_scaler,
                conv_kernel=self.conv_kernel,
                padding=self.padding,
                dropout=self.dropout,
                num_group=self.num_group,
                components=self.layer_components,
                pool_kernel=self.pool_kernel,
                conv_module=RecurrentDoubleConvolution,
            )

        """ Decoder """
        self.decoder = build_decoder(
            conv_layers=self.num_conv_layer,
            conv_layer_scaler=self.conv_layer_scaler,
            components=self.layer_components,
            conv_kernel=self.conv_kernel,
            padding=self.padding,
            sizes=self.patch_sizes,
            num_group=self.num_group,
            deconv_module=self.model,
        )

        """ Final Layer """
        if "3" in self.layer_components:
            self.final_conv_layer = nn.Conv3d(
                in_channels=self.conv_layer_scaler,
                out_channels=self.out_channels,
                kernel_size=1,
            )
        elif "2" in self.layer_components:
            self.final_conv_layer = nn.Conv2d(
                in_channels=self.conv_layer_scaler,
                out_channels=self.out_channels,
                kernel_size=1,
            )

        """ Prediction """
        if self.sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)


class UNet(BasicCNN):
    """
    Implementation of a U-Net model derived from a basic convolutional neural network.

    The U-Net model is widely used in image segmentation tasks. This class is designed to
    handle encoder-decoder architectures for feature extraction and reconstruction. It applies
    the U-Net structure with added flexibility for extensions or customizations as needed in
    specific tasks.

    "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    <https://arxiv.org/pdf/1606.06650.pdf>.
    """

    def __init__(self, model="CNN", **kwargs):
        super(UNet, self).__init__(**kwargs)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network which includes an encoding stage,
        decoding stage, and a final prediction step.

        :param x: Input tensor that will be passed through the encoder, decoder, and
            prediction stages of the network.
        :type x: torch.Tensor
        :return: Output tensor produced after applying the encoder, decoder,
            and the final activation (if prediction is enabled).
        :rtype: torch.Tensor
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
    A residual U-Net (ResUNet) model implementation extending the BasicCNN base class.

    A class structure designed for image segmentation tasks utilizing a Residual U-Net
    architecture. The network is composed of an encoder and a decoder. The encoder
    extracts features, and the decoder reconstructs the features into a segmentation
    mask. It supports flexible customization for model variations through inheritance
    and additional parameters.

    modified of <10.1016/j.isprsjprs.2020.01.013>
    """

    def __init__(self, model="RCNN", **kwargs):
        super(ResUNet, self).__init__(**kwargs)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the network which includes an encoding stage,
        decoding stage, and a final prediction step.

        :param x: Input tensor that will be passed through the encoder, decoder, and
            prediction stages of the network.
        :type x: torch.Tensor
        :return: Output tensor produced after applying the encoder, decoder,
            and the final activation (if prediction is enabled).
        :rtype: torch.Tensor
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
    UNet3Plus is a neural network model based on the U-Net architecture with enhancements
    specifically designed for segmentation tasks. It integrates improvements in both encoder
    and decoder components and optionally provides classification capabilities. This class is
    highly configurable and supports both 2D and 3D inputs.

    The class consists of an encoder that extracts features from the input, a decoder
    that reconstructs the segmentation map from these features, optional classification
    modules, and a final prediction activation layer. Users can configure the number of
    convolutional layers, kernel sizes, pooling operations, dropout, and other settings
    to adapt the model to a wide variety of use cases.

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
        """
        Represents the 3D U-Net++ (UNet3Plus) architecture for segmentation tasks with options
        for classification. The class supports configurable numbers of layers, convolution
        properties, dropout, group normalization, and multiple segmentation and classification
        settings.

        :param in_channels:
            Number of input channels for the network.
        :param out_channels:
            Number of output channels from the network.
        :param sigmoid:
            Whether to use Sigmoid activation for prediction. If False, Softmax activation
            will be used.
        :param num_conv_layer:
            Number of convolutional layers to use.
        :param conv_layer_scaler:
            Scaling factor for the number of feature maps in each convolutional layer.
        :param conv_kernel:
            Size of the convolutional kernels.
        :param padding:
            Amount of padding to apply in convolutional operations.
        :param pool_kernel:
            Size of the pooling kernels.
        :param img_patch_size:
            Size of the initial image patch.
        :param layer_components:
            Indicates the dimensionality (2D or 3D convolution) and whether grouped
            convolution or residual layers are used. For example, "3gcl".
        :param dropout:
            Probability of dropout for the layers. If None, dropout is not applied.
        :param num_group:
            Number of groups for group normalization layers.
        :param prediction:
            Flag to indicate whether the network produces explicit predictions.
        :param classifies:
            Flag to indicate whether the network includes a classification head.
        :param decoder_features:
            Optional additional features for the decoder configuration.
        """
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
        Computes the dot product of two tensors `x` and `x_cls` along specific dimensions.
        The function first reshapes the input tensors to flatten the spatial dimensions
        (H, W) and depth (D) into a single dimension for efficient computation. Then,
        it performs an element-wise dot product of the reshaped tensors. The result is
        reshaped back into the original spatial and depth dimensions of the inputs.

        :param x:
            A tensor of shape (B, N, D, H, W), where B represents the batch size,
            N represents the number of channels or features, D represents the depth,
            and H, W represent spatial dimensions.
        :param x_cls:
            A tensor of shape (B, N, D, H, W), aligned with the same shape as `x`,
            used for element-wise dot product computation.
        :return:
            A tensor of the same shape (B, N, D, H, W) as the input, containing the
            result of the dot product operation computed along specific dimensions.
        """
        B, N, D, H, W = x.shape
        x = x.view(B, N, D * H * W)

        final = torch.einsum("ijk,ijk->ijk", [x, x_cls])
        final = final.view(B, N, D, H, W)

        return final

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass through the network which includes an encoding stage,
        decoding stage, and a final prediction step.

        :param x: Input tensor that will be passed through the encoder, decoder, and
            prediction stages of the network.
        :type x: torch.Tensor
        :return: Output tensor produced after applying the encoder, decoder,
            and the final activation (if prediction is enabled).
        :rtype: torch.Tensor
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
    FNet model for image segmentation.

    This class implements the FNet model, designed for feature extraction and image
    segmentation tasks. It includes an encoder for processing input features, two decoder
    variants (UNet and UNet3+), and a final output layer for generating a segmentation
    map. It supports configurable parameters for convolutional layers, kernel size, dropout,
    padding, pool kernel size, and other hyperparameters. The class can predict either
    sigmoid- or softmax-activated probability masks for segmentation.
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
        """
        Initializes the FNet class, which serves as the main implementation
        for a configurable neural network architecture. This class includes
        several components such as an encoder, decoders, convolutional layers,
        and support for attention-based features. The class allows customization
        of the architecture through input parameters to tailor it for
        specific tasks.

        :param in_channels: Number of input channels for the network.
        :param out_channels: Number of output channels generated by the network.
        :param sigmoid: Determines whether to apply sigmoid activation at
            the final layer, typically used in binary classification problems.
        :param num_conv_layer: Controls the number of convolutional layers
            in the architecture.
        :param conv_layer_scaler: Factor to multiply the number of
            convolutional filters at each layer.
        :param conv_kernel: Kernel size for the convolutional layers.
        :param padding: Amount of padding for convolutional layers
            (e.g., same or valid).
        :param pool_kernel: Kernel size to use for pooling operations.
        :param dropout: Defines the dropout rate applied to intermediate
            layers to prevent overfitting. If None, no dropout is applied.
        :param img_patch_size: Input image patch size used during
            the network's processing pipeline.
        :param layer_components: String specifying layer configurations
            (e.g., "3gcl" could refer to grouped convolutions with attention layers).
        :param num_group: Defines the number of groups for grouped convolutions.
        :param attn_features: Boolean determining whether attention-based
            features are included in the network's architecture.
        :param prediction: Whether the model is in prediction mode, which
            influences components such as the patch size and activation layers.
        """
        super(FNet, self).__init__()
        self.prediction = prediction
        self.encoder, self.decoder_unet, self.decoder_3plus = None, None, None
        self.unet_conv_layer, self.unet3plus_conv_layer, self.final_conv_layer = (
            None,
            None,
            None,
        )
        self.activation = None

        self.patch_sizes = img_patch_size

        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_conv_layer = num_conv_layer
        self.conv_layer_scaler = conv_layer_scaler
        self.conv_kernel = conv_kernel
        self.padding = padding
        self.dropout = dropout
        self.num_group = num_group
        self.layer_components = layer_components
        self.pool_kernel = pool_kernel

        self.attn_features = attn_features
        self.sigmoid = sigmoid

        self.patch_sizes = [img_patch_size]
        self.update_patch_size(img_patch_size, self.prediction)
        self.build_cnn_model()

    def update_patch_size(self, img_patch_size, prediction):
        """
        Updates the image patch sizes and rebuilds decoders with the new configuration.

        This method is responsible for updating the internal patch sizes based on the
        provided `img_patch_size` and constructing new decoders (`decoder_unet` and
        `decoder_3plus`) using the updated patch sizes. Furthermore, it preserves the
        state of the existing decoder models and reloads them into the newly built
        decoders.

        :param img_patch_size: Initial size of the image patch to be used for calculating
            new patch sizes across layers.
        :type img_patch_size: int
        :param prediction: Indicates if the operation is performed during prediction mode.
        :type prediction: bool
        :return: This method does not return a value.
        :rtype: None
        """
        self.prediction = prediction

        self.patch_sizes = [img_patch_size]
        for _ in range(self.num_conv_layer):
            img_patch_size = int(img_patch_size / 2)
            self.patch_sizes.append(img_patch_size)
        self.patch_sizes = list(reversed(self.patch_sizes))[2:]

        if self.decoder_unet is not None:
            states = [self.decoder_unet.state_dict(), self.decoder_3plus.state_dict()]

            """ Decoder """
            self.decoder_unet = build_decoder(
                conv_layers=self.num_conv_layer,
                conv_layer_scaler=self.conv_layer_scaler,
                components=self.layer_components,
                conv_kernel=self.conv_kernel,
                padding=self.padding,
                sizes=self.patch_sizes,
                num_group=self.num_group,
            )
            self.decoder_3plus = build_decoder(
                conv_layers=self.num_conv_layer,
                conv_layer_scaler=self.conv_layer_scaler,
                components=self.layer_components,
                conv_kernel=self.conv_kernel,
                padding=self.padding,
                sizes=self.patch_sizes,
                num_group=self.num_group,
                deconv_module="unet3plus",
                attn_features=self.attn_features,
            )
            self.decoder_unet.load_state_dict(states[0])
            self.decoder_3plus.load_state_dict(states[1])

            states = None

    def build_cnn_model(self):
        """
        Build the CNN model by constructing encoder, decoder, and final prediction layers.
        """
        """ Encoder """
        self.encoder = build_encoder(
            in_ch=self.in_channels,
            conv_layers=self.num_conv_layer,
            conv_layer_scaler=self.conv_layer_scaler,
            conv_kernel=self.conv_kernel,
            padding=self.padding,
            dropout=self.dropout,
            num_group=self.num_group,
            components=self.layer_components,
            pool_kernel=self.pool_kernel,
            conv_module=DoubleConvolution,
            attn_features=self.attn_features,
        )

        """ Decoder """
        self.decoder_unet = build_decoder(
            conv_layers=self.num_conv_layer,
            conv_layer_scaler=self.conv_layer_scaler,
            components=self.layer_components,
            conv_kernel=self.conv_kernel,
            padding=self.padding,
            sizes=self.patch_sizes,
            num_group=self.num_group,
        )
        self.decoder_3plus = build_decoder(
            conv_layers=self.num_conv_layer,
            conv_layer_scaler=self.conv_layer_scaler,
            components=self.layer_components,
            conv_kernel=self.conv_kernel,
            padding=self.padding,
            sizes=self.patch_sizes,
            num_group=self.num_group,
            deconv_module="unet3plus",
            attn_features=self.attn_features,
        )

        """ Final Layer """
        if "3" in self.layer_components:
            self.unet_conv_layer = nn.Conv3d(
                in_channels=self.conv_layer_scaler,
                out_channels=self.conv_layer_scaler,
                kernel_size=1,
            )
            self.unet3plus_conv_layer = nn.Conv3d(
                in_channels=self.conv_layer_scaler,
                out_channels=self.conv_layer_scaler,
                kernel_size=1,
            )

            self.final_conv_layer = nn.Conv3d(
                in_channels=self.conv_layer_scaler * 2,
                out_channels=self.out_channels,
                kernel_size=1,
            )
        elif "2" in self.layer_components:
            self.unet_conv_layer = nn.Conv2d(
                in_channels=self.conv_layer_scaler,
                out_channels=self.conv_layer_scaler,
                kernel_size=1,
            )
            self.unet3plus_conv_layer = nn.Conv2d(
                in_channels=self.conv_layer_scaler,
                out_channels=self.conv_layer_scaler,
                kernel_size=1,
            )

            self.final_conv_layer = nn.Conv2d(
                in_channels=self.conv_layer_scaler * 2,
                out_channels=self.out_channels,
                kernel_size=1,
            )

        """ Prediction """
        if self.sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        """
        Performs the forward pass through the network which includes an encoding stage,
        decoding stage, and a final prediction step.

        :param x: Input tensor that will be passed through the encoder, decoder, and
            prediction stages of the network.
        :type x: torch.Tensor
        :return: Output tensor produced after applying the encoder, decoder,
            and the final activation (if prediction is enabled).
        :rtype: torch.Tensor
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
