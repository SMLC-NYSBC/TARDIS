from typing import Optional

import torch
from tardis.spindletorch.utils.build_network import build_network


class TestNetwork3D:
    image_sizes = [32, 64, 96]
    conv_multiplayers = [32, 64]

    def unet_3d(self,
                image_size: int,
                layer_no: int,
                conv_multiplayer: int,
                dropout: Optional[float] = None):
        return build_network(network_type='unet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_kernel=3,
                             conv_padding=1,
                             maxpool_kernel=2,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='3gcl',
                             no_groups=8,
                             prediction=False)

    def unet_2d(self,
                image_size: int,
                layer_no: int,
                conv_multiplayer: int,
                dropout: Optional[float] = None):
        return build_network(network_type='unet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             conv_kernel=3,
                             conv_padding=1,
                             maxpool_kernel=2,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='2gcl',
                             no_groups=8,
                             prediction=False)

    def resunet_3d(self,
                   image_size: int,
                   layer_no: int,
                   conv_multiplayer: int,
                   dropout: Optional[float] = None):
        return build_network(network_type='resunet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='3gcl',
                             no_groups=8,
                             prediction=False)

    def resunet_2d(self,
                   image_size: int,
                   layer_no: int,
                   conv_multiplayer: int,
                   dropout: Optional[float] = None):
        return build_network(network_type='resunet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='2gcl',
                             no_groups=8,
                             prediction=False)

    def unet3plus_3d(self,
                     image_size: int,
                     layer_no: int,
                     conv_multiplayer: int,
                     dropout: Optional[float] = None):
        return build_network(network_type='unet3plus',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='3gcl',
                             no_groups=8,
                             prediction=False)

    def unet3plus_2d(self,
                     image_size: int,
                     layer_no: int,
                     conv_multiplayer: int,
                     dropout: Optional[float] = None):
        return build_network(network_type='unet3plus',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='2gcl',
                             no_groups=8,
                             prediction=False)

    def big_unet_2d(self,
                    image_size: int,
                    layer_no: int,
                    conv_multiplayer: int,
                    dropout: Optional[float] = None):
        return build_network(network_type='big_unet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='2gcl',
                             no_groups=8,
                             prediction=False)

    def big_unet_3d(self,
                    image_size: int,
                    layer_no: int,
                    conv_multiplayer: int,
                    dropout: Optional[float] = None):
        return build_network(network_type='big_unet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='3gcl',
                             no_groups=8,
                             prediction=False)

    def wnet_2d(self,
                image_size: int,
                layer_no: int,
                conv_multiplayer: int,
                dropout: Optional[float] = None):
        return build_network(network_type='wnet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='2gcl',
                             no_groups=8,
                             prediction=False)

    def wnet_3d(self,
                image_size: int,
                layer_no: int,
                conv_multiplayer: int,
                dropout: Optional[float] = None):
        return build_network(network_type='wnet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             no_conv_layers=layer_no,
                             conv_multiplayer=conv_multiplayer,
                             layer_components='3gcl',
                             no_groups=8,
                             prediction=False)

    def test_unet3d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.unet_3d(image_size=i,
                                  layer_no=5,
                                  conv_multiplayer=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet2d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.unet_2d(image_size=i,
                                  layer_no=5,
                                  conv_multiplayer=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_resunet_3d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.resunet_3d(image_size=i,
                                     layer_no=5,
                                     conv_multiplayer=j,
                                     dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_resunet_2d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.resunet_2d(image_size=i,
                                     layer_no=5,
                                     conv_multiplayer=j,
                                     dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet3plus_2d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.unet3plus_2d(image_size=i,
                                       layer_no=5,
                                       conv_multiplayer=j,
                                       dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet3plus_3d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.unet3plus_3d(image_size=i,
                                       layer_no=5,
                                       conv_multiplayer=j,
                                       dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_big_unet_2d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.big_unet_2d(image_size=i,
                                      layer_no=5,
                                      conv_multiplayer=j,
                                      dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_big_unet_3d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.big_unet_3d(image_size=i,
                                      layer_no=5,
                                      conv_multiplayer=j,
                                      dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_wnet_2d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.wnet_2d(image_size=i,
                                  layer_no=5,
                                  conv_multiplayer=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_wnet_3d(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.wnet_3d(image_size=i,
                                  layer_no=5,
                                  conv_multiplayer=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)
