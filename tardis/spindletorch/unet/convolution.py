from math import sqrt
from typing import Optional

import torch
import torch.nn as nn


class GeLU(nn.Module):
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        tanh: hyperbolic tangent value for GeLU
        x: torch input for activation.
    """
    def __init__(self,
                 tanh: Optional[float] = None):
        super(GeLU, self).__init__()

        if tanh is not None:
            self.tanh = sqrt(tanh)
        else:
            self.tanh = 1.41421356237

    def forward(self,
                x: torch.Tensor):
        return x * 0.5 * (1.0 + torch.erf(x / self.tanh))


def convolution(in_ch: int,
                out_ch: int,
                components: str,
                kernel: int or tuple,
                padding: int or tuple,
                no_group=None):
    """
    Build an convolution block with a specified components and order:
    dimension (2 or 3), conv (c), ReLu (r), LeakyReLu (l), GeLu (e),
    GroupNorm (g), BatchNorm (b).

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        components: Components that are used for conv. block
        kernel: Kernel size for the convolution
        padding: Padding size for the convolution
        no_group: No. of groups for the nn.GroupNorm.
            None -> if nn.GroupNorm is not used
    """

    modules = []
    conv = False

    """Build single CNN block"""
    for letter in components:
        if "c" in letter:
            conv = True

            if '3' in components:
                if 'g' in components or 'b' in components:
                    modules.append(("Conv3D", nn.Conv3d(in_channels=in_ch,
                                                        out_channels=out_ch,
                                                        kernel_size=kernel,
                                                        padding=padding,
                                                        bias=False)))
                else:
                    modules.append(("Conv3D", nn.Conv3d(in_channels=in_ch,
                                                        out_channels=out_ch,
                                                        kernel_size=kernel,
                                                        padding=padding,
                                                        bias=True)))
            if '2' in components:
                if 'g' in components or 'b' in components:
                    modules.append(("Conv2D", nn.Conv2d(in_channels=in_ch,
                                                        out_channels=out_ch,
                                                        kernel_size=kernel,
                                                        padding=padding,
                                                        bias=False)))
                else:
                    modules.append(("Conv2D", nn.Conv2d(in_channels=in_ch,
                                                        out_channels=out_ch,
                                                        kernel_size=kernel,
                                                        padding=padding,
                                                        bias=True)))

        if "g" == letter:
            assert no_group is not None, \
                'Number of group is required if nn.GroupNorm is used.'

            if no_group > in_ch:
                no_group = 1
            if conv:
                modules.append(("GroupNorm1", nn.GroupNorm(num_groups=no_group,
                                                           num_channels=out_ch)))
            else:
                modules.append(("GroupNorm2", nn.GroupNorm(num_groups=no_group,
                                                           num_channels=in_ch)))

        if "b" == letter:
            if '3' in components:
                if conv:
                    modules.append(("BatchNorm1", nn.BatchNorm3d(out_ch)))
                else:
                    modules.append(("BatchNorm2", nn.BatchNorm3d(in_ch)))
            if '2' in components:
                if conv:
                    modules.append(("BatchNorm1", nn.BatchNorm2d(out_ch)))
                else:
                    modules.append(("BatchNorm2", nn.BatchNorm2d(in_ch)))

        if "r" in letter:
            modules.append(("ReLu", nn.ReLU()))
        if "l" in letter:
            modules.append(("LeakyReLu", nn.LeakyReLU(negative_slope=0.1)))
        if 'e' in letter:
            modules.append(('GeLU', GeLU()))

    return modules


class SingleConvolution(nn.Sequential):
    """
    SINGLE 3D CONVOLUTION BLOCK

    Output single convolution composed of conv, normalization and relu in order
    defined by components variable.

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        kernel: Kernel size for the convolution
        padding: Padding size for the convolution
        components: Components for the convolution block
        no_group: No of groups if nn.GroupNorm is used
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 components: str,
                 kernel: int or tuple,
                 padding: int or tuple,
                 no_group=None,
                 block_type=None):
        super(SingleConvolution, self).__init__()
        if '3' in components:
            conv3d = convolution(in_ch=in_ch,
                                 out_ch=out_ch,
                                 components=components,
                                 kernel=kernel,
                                 padding=padding,
                                 no_group=no_group)

            for name, module in conv3d:
                self.add_module(name, module)

        if '2' in components:
            conv2d = convolution(in_ch=in_ch,
                                 out_ch=out_ch,
                                 components=components,
                                 kernel=kernel,
                                 padding=padding,
                                 no_group=no_group)

            for name, module in conv2d:
                self.add_module(name, module)


class DoubleConvolution(nn.Sequential):
    """
    DOUBLE CONVOLUTION BLOCK

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        block_type: Define encode or decoder path e.g.
            'encoder': Encoder convolution path
            'decoder': Decoder convolution path
        kernel: Kernel size for the convolution
        padding: Padding size for the convolution
        components: Components for the convolution block
        no_group: No of groups if nn.GroupNorm is used
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 block_type: str,
                 kernel: int or tuple,
                 padding: int or tuple,
                 components="cgr",
                 no_group=None):
        super(DoubleConvolution, self).__init__()

        # Define in and out channels for 1st and 2nd convolutions
        assert block_type in ["encoder", "decoder"], \
            'Only "encoder", "decoder block type is supported.'

        if block_type == "encoder":
            conv1_in_ch, conv1_out_ch = in_ch, out_ch // 2
            conv2_in_ch, conv2_out_ch = conv1_out_ch, out_ch

        if block_type == "decoder":
            conv1_in_ch, conv1_out_ch = in_ch, out_ch
            conv2_in_ch, conv2_out_ch = out_ch, out_ch

        self.add_module("DoubleConv1",
                        SingleConvolution(in_ch=conv1_in_ch,
                                          out_ch=conv1_out_ch,
                                          kernel=kernel,
                                          padding=padding,
                                          components=components,
                                          no_group=no_group))

        self.add_module("DoubleConv2",
                        SingleConvolution(in_ch=conv2_in_ch,
                                          out_ch=conv2_out_ch,
                                          components=components,
                                          kernel=kernel,
                                          padding=padding,
                                          no_group=no_group))


class RecurrentDoubleConvolution(nn.Module):
    """
    RECURRENT DOUBLE CONVOLUTION BLOCK

    Args:
        in_ch: Number of input channels
        out_ch: Number of output channels
        block_type: Define encode or decoder path e.g.
            'encoder': Encoder convolution path
            'decoder': Decoder convolution path
        kernel: Kernel size for the convolution
        padding: Padding size for the convolution
        components: Components for the convolution block
        no_group: No of groups if nn.GroupNorm is used
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 block_type: str,
                 kernel: int or tuple,
                 padding: int or tuple,
                 components="cgr",
                 no_group=None):
        super(RecurrentDoubleConvolution, self).__init__()

        # Define in and out channels for 1st and 2nd convolutions
        assert block_type in ["encoder", "decoder"], \
            'Only "encoder", "decoder block type is supported.'

        if block_type == "encoder":
            conv1_in_ch, conv1_out_ch = in_ch, out_ch // 2
            conv2_in_ch, conv2_out_ch = conv1_out_ch, out_ch

        if block_type == "decoder":
            conv1_in_ch, conv1_out_ch = in_ch, out_ch
            conv2_in_ch, conv2_out_ch = out_ch, out_ch

        self.conv1 = SingleConvolution(in_ch=conv1_in_ch,
                                       out_ch=conv1_out_ch,
                                       kernel=kernel,
                                       padding=padding,
                                       components=components,
                                       no_group=no_group)

        self.conv2 = SingleConvolution(in_ch=conv2_in_ch,
                                       out_ch=conv2_out_ch,
                                       components=components,
                                       kernel=kernel,
                                       padding=padding,
                                       no_group=no_group)

        self.conv3 = SingleConvolution(in_ch=conv2_out_ch,
                                       out_ch=conv2_out_ch,
                                       components='c',
                                       kernel=kernel,
                                       padding=padding,
                                       no_group=no_group)
        if 'l' in components:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'r' in components:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self,
                x: torch.Tensor):
        out = self.conv1(x)
        out = self.conv2(out)
        residual = out

        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)

        return out
