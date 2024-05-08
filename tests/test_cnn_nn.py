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

from tardis_em.cnn.cnn import build_cnn_network

structure = {
    "in_channel": 1,
    "out_channel": 1,
    "img_size": 64,
    "dropout": None,
    "conv_kernel": 3,
    "conv_padding": 1,
    "maxpool_kernel": 2,
    "num_group": 1,
    "num_conv_layers": 5,
    "conv_scaler": 32,
    "classification": False,
    "layer_components": "3gcl",
}


def unet_3d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "3gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="unet", structure=structure, img_size=image_size, prediction=False
    )


def unet_2d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "2gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="unet", structure=structure, img_size=image_size, prediction=False
    )


def resunet_3d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "3gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="resunet",
        structure=structure,
        img_size=image_size,
        prediction=False,
    )


def resunet_2d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "2gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="resunet",
        structure=structure,
        img_size=image_size,
        prediction=False,
    )


def unet3plus_3d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "3gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="unet3plus",
        structure=structure,
        img_size=image_size,
        prediction=False,
    )


def unet3plus_2d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "2gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="unet3plus",
        structure=structure,
        img_size=image_size,
        prediction=False,
    )


def fnet_3d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "3gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="fnet", structure=structure, img_size=image_size, prediction=False
    )


def fnet_2d(
    image_size: int, layer_num: int, conv_scaler: int, dropout: Optional[float] = None
):
    structure.update({"layer_components": "2gcl"})
    structure.update({"num_conv_layers": layer_num})
    structure.update({"conv_scaler": conv_scaler})
    structure.update({"dropout": dropout})
    return build_cnn_network(
        network_type="fnet", structure=structure, img_size=image_size, prediction=False
    )


class TestNetwork3D:
    image_sizes = [16, 32, 64, 96]
    conv_scaler = [4, 8, 16, 32]

    def test_unet3d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = unet_3d(image_size=i, layer_num=5, conv_scaler=j, dropout=None)

                with torch.no_grad():
                    nn(input)

    def test_unet2d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = unet_2d(image_size=i, layer_num=5, conv_scaler=j, dropout=None)

                with torch.no_grad():
                    nn(input)

    def test_resunet_3d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = resunet_3d(image_size=i, layer_num=5, conv_scaler=j, dropout=None)

                with torch.no_grad():
                    nn(input)

    def test_resunet_2d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = resunet_2d(image_size=i, layer_num=5, conv_scaler=j, dropout=None)

                with torch.no_grad():
                    nn(input)

    # def test_unet3plus_2d(self):
    #     for i in self.image_sizes:
    #         for j in self.conv_scaler:
    #             # Batch x Channel x Z x Y x X
    #             input = torch.rand((1, 1, i, i))
    #
    #             nn = unet3plus_2d(
    #                 image_size=i, layer_num=5, conv_scaler=j, dropout=None
    #             )
    #
    #             with torch.no_grad():
    #                 nn(input)
    #
    # def test_unet3plus_3d(self):
    #     for i in self.image_sizes:
    #         for j in self.conv_scaler:
    #             # Batch x Channel x Z x Y x X
    #             input = torch.rand((1, 1, i, i, i))
    #
    #             nn = unet3plus_3d(
    #                 image_size=i, layer_num=5, conv_scaler=j, dropout=None
    #             )
    #
    #             with torch.no_grad():
    #                 nn(input)

    # def test_fnet_2d(self):
    #     for i in self.image_sizes:
    #         for j in self.conv_scaler:
    #             # Batch x Channel x Z x Y x X
    #             input = torch.rand((1, 1, i, i))
    #
    #             nn = fnet_2d(image_size=i, layer_num=5, conv_scaler=j, dropout=None)
    #
    #             with torch.no_grad():
    #                 nn(input)
    #
    # def test_fnet_3d(self):
    #     for i in self.image_sizes:
    #         for j in self.conv_scaler:
    #             # Batch x Channel x Z x Y x X
    #             input = torch.rand((1, 1, i, i, i))
    #
    #             nn = fnet_3d(image_size=i, layer_num=5, conv_scaler=j, dropout=None)
    #
    #             with torch.no_grad():
    #                 nn(input)
