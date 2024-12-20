#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from math import sqrt
from typing import Optional

import torch
import torch.nn as nn

from tardis_em.utils.errors import TardisError


class GeLU(nn.Module):
    """
    Applies the Gaussian Error Linear Unit (GeLU) activation function.

    The GeLU activation function is a smooth approximation to the ReLU activation function.
    This implementation provides an optional `tanh` parameter that can be used to scale the
    input argument to the standard mathematical error function (erf). It is primarily used
    in neural networks to introduce non-linearity.
    """

    def __init__(self, tanh: Optional[float] = None):
        """
        Represents the Gaussian Error Linear Unit (GeLU) function used as an activation function
        in machine learning models. This implementation contains an optional tangent hyperbolic
        approximation factor that alters the behavior of the GeLU computation.

        :param tanh: Optional tangent hyperbolic approximation factor. If provided, its square
            root is calculated and stored as an attribute.
        :type tanh: float or None
        """
        super(GeLU, self).__init__()

        if tanh is not None:
            self.tanh = sqrt(tanh)
        else:
            self.tanh = 1.4142135623730951

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a transformation to the input tensor using the hyperbolic tangent error function,
        scaling the result accordingly. This method is commonly utilized in machine learning models
        and functions by smoothing input data values into a specified scaling range.

        :param x: Input tensor on which the transformation is applied.
        :type x: torch.Tensor

        :return: Transformed tensor after applying the scaled error function operation.
        :rtype: torch.Tensor
        """
        return x * 0.5 * (1.0 + torch.erf(x / self.tanh))


def convolution(
    in_ch: int,
    out_ch: int,
    components: str,
    kernel: int or tuple,
    padding: int or tuple,
    num_group=None,
) -> list:
    """
    Builds a neural network block by assembling components specified in the given
    string. This function enables the construction of customizable CNN layers
    based on the input parameters and component configurations. The components
    can consist of various operations such as convolutions (2D or 3D), normalization
    techniques (GroupNorm, BatchNorm), and activation functions (ReLU, LeakyReLU,
    GeLU, PReLU). These components are added sequentially to the module list based
    on their order in the `components` string.

    :param in_ch: Number of input channels for the convolutional layers.
    :type in_ch: int
    :param out_ch: Number of output channels for the convolutional layers.
    :type out_ch: int
    :param components: A string that specifies the sequence of operations for the
                       CNN block. Each character or group represents a specific
                       operation such as convolution, normalization, or activation.
    :type components: str
    :param kernel: Kernel size for the convolution. Can be an integer or a tuple
                   representing the size.
    :type kernel: int or tuple
    :param padding: Padding size for the convolution. Can be an integer or a tuple
                    representing the size.
    :type padding: int or tuple
    :param num_group: Number of groups for Group Normalization. Required if
                      GroupNorm is included in the components. Defaults to None.
    :type num_group: int, optional

    :return: A list of tuples representing the layers in the CNN block.
             Each tuple contains a layer name and its corresponding module.
    :rtype: list
    """
    modules = []
    conv = False

    """Build CNN block"""
    for letter in components:
        """Add Convolution"""
        if "c" in letter:
            conv = True

            # Add 3DConv
            if "3" in components:
                if "g" in components or "b" in components:
                    modules.append(
                        (
                            "Conv3D",
                            nn.Conv3d(
                                in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=kernel,
                                padding=padding,
                                bias=False,
                            ),
                        )
                    )
                else:
                    modules.append(
                        (
                            "Conv3D",
                            nn.Conv3d(
                                in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=kernel,
                                padding=padding,
                            ),
                        )
                    )
            # Add 2DConv
            if "2" in components:
                if "g" in components or "b" in components:
                    modules.append(
                        (
                            "Conv2D",
                            nn.Conv2d(
                                in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=kernel,
                                padding=padding,
                                bias=False,
                            ),
                        )
                    )
                else:
                    modules.append(
                        (
                            "Conv2D",
                            nn.Conv2d(
                                in_channels=in_ch,
                                out_channels=out_ch,
                                kernel_size=kernel,
                                padding=padding,
                            ),
                        )
                    )
        """Add GroupNorm"""
        if "g" == letter:
            if num_group is None:
                TardisError(
                    "142",
                    "tardis_em/cnn/model/convolution.py",
                    "Number of group is required if nn.GroupNorm is used.",
                )

            if num_group > in_ch:
                num_group = 1
            if conv:
                modules.append(
                    (
                        "GroupNorm1",
                        nn.GroupNorm(num_groups=num_group, num_channels=out_ch),
                    )
                )
            else:
                modules.append(
                    (
                        "GroupNorm2",
                        nn.GroupNorm(num_groups=num_group, num_channels=in_ch),
                    )
                )
        """Add BatchNorm"""
        if "b" == letter:
            if "3" in components:
                if conv:
                    modules.append(("BatchNorm1", nn.BatchNorm3d(out_ch)))
                else:
                    modules.append(("BatchNorm2", nn.BatchNorm3d(in_ch)))
            if "2" in components:
                if conv:
                    modules.append(("BatchNorm1", nn.BatchNorm2d(out_ch)))
                else:
                    modules.append(("BatchNorm2", nn.BatchNorm2d(in_ch)))
        """Add ReLu"""
        if "r" in letter:
            modules.append(("ReLu", nn.ReLU()))
        """Add LeakyReLu"""
        if "l" in letter:
            modules.append(("LeakyReLu", nn.LeakyReLU(negative_slope=0.1)))
        """Add GeLu"""
        if "e" in letter:
            modules.append(("GeLU", GeLU()))
        if "p" in letter:
            modules.append(("PReLU", nn.PReLU(num_parameters=out_ch)))

    return modules


class SingleConvolution(nn.Sequential):
    """
    Represents a sequential layer that contains a single convolution operation.

    This class is a part of a neural network building process, specifically for
    performing either 2D or 3D convolutional operations. It inherits from
    `nn.Sequential` to combine multiple convolutional modules into a sequence
    based on the specified dimensionality of the operation. It dynamically
    constructs convolutional layers considering the input parameters.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        components: str,
        kernel: int or tuple,
        padding: int or tuple,
        num_group=None,
        block_type="any",
    ):
        """
        Represents a single convolution operation, supporting 2D or 3D convolution
        based on the input arguments. The class dynamically constructs and assigns
        the convolution modules during initialization for efficient and modular
        integration.

        :param in_ch: Number of input channels for the convolution operation.
        :type in_ch: int
        :param out_ch: Number of output channels from the convolution operation.
        :type out_ch: int
        :param components: Specification for the components to be utilized,
            determining if 2D or 3D convolution is applied. It should include
            '2' for 2D and '3' for 3D components processing.
        :type components: str
        :param kernel: Size of the convolution kernel. Accepts either an int or a
            tuple reflecting the dimensionality of the kernel.
        :type kernel: int or tuple
        :param padding: Padding value applied to the input, configurable by
            integer or tuple for kernel dimension matching.
        :type padding: int or tuple
        :param num_group: Optional grouping parameter for grouped convolution,
            defaults to None implying standard convolution behavior.
        :type num_group: int, optional
        :param block_type: An optional string parameter to define the specific
            block type functionality. Defaults to "any".
        :type block_type: str, optional
        """
        super(SingleConvolution, self).__init__()

        """Build single Conv3D"""
        if "3" in components:
            conv3d = convolution(
                in_ch=in_ch,
                out_ch=out_ch,
                components=components,
                kernel=kernel,
                padding=padding,
                num_group=num_group,
            )

            for name, module in conv3d:
                self.add_module(name, module)

        """Build single Conv2D"""
        if "2" in components:
            conv2d = convolution(
                in_ch=in_ch,
                out_ch=out_ch,
                components=components,
                kernel=kernel,
                padding=padding,
                num_group=num_group,
            )

            for name, module in conv2d:
                self.add_module(name, module)


class DoubleConvolution(nn.Sequential):
    """
    Implements a double convolutional block feature in a neural network.

    This class is a specialized neural network module implementing a double
    convolutional operation, designed either as an encoder block for reducing
    spatial resolution while increasing the number of features or as a decoder
    block for recovering spatial resolution with retained channel properties.
    It supports two block types - "encoder" and "decoder", leveraging the
    flexibility of defining kernel size, padding, and customizable components.
    The concept of group convolutions is enabled for enhanced modularity via
    `num_group`.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        block_type: str,
        kernel: int or tuple,
        padding: int or tuple,
        components="cgr",
        num_group=None,
    ):
        """
        This class initializes a double convolution block by chaining two convolutional layers
        using the specified input and output channel characteristics, block type, kernel properties,
        and other optional configurations. It dynamically computes the input and output channels
        for each convolution layer based on the block type.

        :param in_ch: The number of input channels for the convolution block.
        :param out_ch: The number of output channels for the convolution block.
        :param block_type: Specifies the type of block, either "encoder" or "decoder".
        :param kernel: The kernel size(s) for the convolution layers. Can be an integer or a tuple.
        :param padding: The padding size(s) for the convolution layers. Can be an integer or a tuple.
        :param components: Components to include in the convolution (default: "cgr").
        :param num_group: The number of groups for grouped convolutions (optional).

        :raises TardisError: Raised if the block type is not "encoder" or "decoder".
        """
        super(DoubleConvolution, self).__init__()

        # Define in and out channels for 1st and 2nd convolutions
        if block_type not in ["encoder", "decoder"]:
            TardisError(
                "143",
                "tardis_em/cnn/model/convolution.py",
                'Only "encoder", decoder block type is supported.',
            )

        """Calculate in and out channels for double convolution"""
        if block_type == "encoder":
            conv1_in_ch, conv1_out_ch = in_ch, out_ch // 2
            conv2_in_ch, conv2_out_ch = conv1_out_ch, out_ch
        else:
            conv1_in_ch, conv1_out_ch = in_ch, out_ch
            conv2_in_ch, conv2_out_ch = out_ch, out_ch

        """Build double convolution"""
        self.add_module(
            "DoubleConv1",
            SingleConvolution(
                in_ch=conv1_in_ch,
                out_ch=conv1_out_ch,
                kernel=kernel,
                padding=padding,
                components=components,
                num_group=num_group,
            ),
        )

        self.add_module(
            "DoubleConv2",
            SingleConvolution(
                in_ch=conv2_in_ch,
                out_ch=conv2_out_ch,
                components=components,
                kernel=kernel,
                padding=padding,
                num_group=num_group,
            ),
        )


class RecurrentDoubleConvolution(nn.Module):
    """
    Defines the RecurrentDoubleConvolution class, which implements a customized
    recurrent double convolution block designed for encoder and decoder
    operations in convolutional neural networks (CNN). The block consists of
    three consecutive single convolution operations with different configurations.
    It incorporates residual connections and supports various non-linearity
    components such as LeakyReLU, ReLU, and PReLU, making it suitable for
    flexible deep learning architectures.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        block_type: str,
        kernel: int or tuple,
        padding: int or tuple,
        components="cgr",
        num_group=None,
    ):
        """
        Represents a recurrent double convolution block used in constructing
        neural network architectures. This block can be configured as either
        an encoder or decoder block, transforming input feature maps into
        output feature maps through a sequence of convolutions and optional
        non-linearity.

        The structure adapts based on the block type to maintain flexibility
        for various deep learning tasks, such as image reconstruction or
        semantic segmentation.

        :param in_ch: Number of input channels.
        :type in_ch: int
        :param out_ch: Number of output channels.
        :type out_ch: int
        :param block_type: Specifies the type of convolution block ('encoder' or 'decoder').
        :type block_type: str
        :param kernel: Kernel size for the convolutional layers.
        :type kernel: int or tuple
        :param padding: Padding value for the convolutional layers.
        :type padding: int or tuple
        :param components: Components for the convolution operations ('c', 'g', 'r' or their combinations).
                           Defaults to "cgr".
        :type components: str
        :param num_group: Number of groups for group convolution. Defaults to None.
        :type num_group: int, optional
        :raises ValueError: If block_type is not 'encoder' or 'decoder'.
        """
        super(RecurrentDoubleConvolution, self).__init__()

        # Define in and out channels for 1st and 2nd convolutions
        if block_type not in ["encoder", "decoder"]:
            TardisError(
                "143",
                "tardis_em/cnn/model/convolution.py",
                'Only "encoder", decoder block type is supported.',
            )

        """Calculate in and out channels for double convolution"""
        if block_type == "encoder":
            conv1_in_ch, conv1_out_ch = in_ch, out_ch // 2
            conv2_in_ch, conv2_out_ch = conv1_out_ch, out_ch
        else:
            conv1_in_ch, conv1_out_ch = in_ch, out_ch
            conv2_in_ch, conv2_out_ch = out_ch, out_ch

        """Build RCNN block"""
        self.conv1 = SingleConvolution(
            in_ch=conv1_in_ch,
            out_ch=conv1_out_ch,
            kernel=kernel,
            padding=padding,
            components=components,
            num_group=num_group,
        )

        self.conv2 = SingleConvolution(
            in_ch=conv2_in_ch,
            out_ch=conv2_out_ch,
            components=components,
            kernel=kernel,
            padding=padding,
            num_group=num_group,
        )

        self.conv3 = SingleConvolution(
            in_ch=conv2_out_ch,
            out_ch=conv2_out_ch,
            components="c",
            kernel=kernel,
            padding=padding,
            num_group=num_group,
        )
        if "l" in components:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif "r" in components:
            self.non_linearity = nn.ReLU(inplace=True)
        elif "p" in components:
            self.non_linearity = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes input tensor through multiple convolutional layers and applies a
        non-linearity function. Implements a residual connection between intermediate
        convolutions and the final output computation.

        :param x: Input tensor to process through the defined convolutional layers.
        :type x: torch.Tensor

        :return: Transformed tensor after applying convolutional layers, residual
            connection, and non-linearity.
        :rtype: torch.Tensor
        """
        out = self.conv1(x)
        out = self.conv2(out)
        residual = out

        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out
