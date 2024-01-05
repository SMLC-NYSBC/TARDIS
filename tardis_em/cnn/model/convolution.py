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
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        tanh (float): hyperbolic tangent value for GeLU
    """

    def __init__(self, tanh: Optional[float] = None):
        super(GeLU, self).__init__()

        if tanh is not None:
            self.tanh = sqrt(tanh)
        else:
            self.tanh = 1.4142135623730951

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward GeLu transformation.

        Args:
            x (torch.Tensor): Image tensor to transform.

        Returns:
            torch.Tensor: Transformed image tensor.
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
    Customizable convolution block builder.

    Build an convolution block with a specified components and order:
    dimension (2 or 3), conv (c),
    ReLu (r), LeakyReLu (l), GeLu (e), PReLu (p),
    GroupNorm (g), BatchNorm (b).

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        components (str): Components that are used for conv. block.
        kernel (int, tuple): Kernel size for the convolution.
        padding (int, tuple): Padding size for the convolution.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.

    Returns:
        list: Ordered nn.Module list.
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
    STANDARD SINGLE 3D CONVOLUTION BLOCK

    Output single convolution composed of conv, normalization and relu in order
    defined by components variable.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        block_type (str): Define encode or decoder path e.g.
            - 'encoder': Encoder convolution path
            - 'decoder': Decoder convolution path
        components (str): Components that are used for conv. block.
        kernel (int, tuple): Kernel size for the convolution.
        padding (int, tuple): Padding size for the convolution.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.
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
    DOUBLE CONVOLUTION BLOCK

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        block_type (str): Define encode or decoder path e.g.
            - 'encoder': Encoder convolution path
            - 'decoder': Decoder convolution path
        components (str): Components that are used for conv. block.
        kernel (int, tuple): Kernel size for the convolution.
        padding (int, tuple): Padding size for the convolution.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.
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
    RECURRENT DOUBLE CONVOLUTION BLOCK

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        block_type (str): Define encode or decoder path e.g.
            - 'encoder': Encoder convolution path
            - 'decoder': Decoder convolution path
        components (str): Components that are used for conv. block.
        kernel (int, tuple): Kernel size for the convolution.
        padding (int, tuple): Padding size for the convolution.
        num_group (int): Num. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used.
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
        Forward function for customized convolution
        Args:
            x: Image patch.

        Returns:
            torch.Tensor: Up or down convoluted image patch.
        """
        out = self.conv1(x)
        out = self.conv2(out)
        residual = out

        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out
