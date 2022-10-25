from typing import Optional

import torch
from tardis_dev.spindletorch.utils.build_network import build_network


class TestNetwork3D:
    image_sizes = [16, 32]
    conv_scaler = [4, 8, 16]

    def unet_3d(self,
                image_size: int,
                layer_num: int,
                conv_scaler: int,
                dropout: Optional[float] = None):
        return build_network(network_type='unet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_kernel=3,
                             conv_padding=1,
                             maxpool_kernel=2,
                             conv_scaler=conv_scaler,
                             layer_components='3gcl',
                             num_group=8,
                             prediction=False)

    def unet_2d(self,
                image_size: int,
                layer_num: int,
                conv_scaler: int,
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
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='2gcl',
                             num_group=8,
                             prediction=False)

    def resunet_3d(self,
                   image_size: int,
                   layer_num: int,
                   conv_scaler: int,
                   dropout: Optional[float] = None):
        return build_network(network_type='resunet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='3gcl',
                             num_group=8,
                             prediction=False)

    def resunet_2d(self,
                   image_size: int,
                   layer_num: int,
                   conv_scaler: int,
                   dropout: Optional[float] = None):
        return build_network(network_type='resunet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='2gcl',
                             num_group=8,
                             prediction=False)

    def unet3plus_3d(self,
                     image_size: int,
                     layer_num: int,
                     conv_scaler: int,
                     dropout: Optional[float] = None):
        return build_network(network_type='unet3plus',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='3gcl',
                             num_group=8,
                             prediction=False)

    def unet3plus_2d(self,
                     image_size: int,
                     layer_num: int,
                     conv_scaler: int,
                     dropout: Optional[float] = None):
        return build_network(network_type='unet3plus',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='2gcl',
                             num_group=8,
                             prediction=False)

    def fnet_2d(self,
                image_size: int,
                layer_num: int,
                conv_scaler: int,
                dropout: Optional[float] = None):
        return build_network(network_type='fnet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='2gcl',
                             num_group=8,
                             prediction=False)

    def fnet_3d(self,
                image_size: int,
                layer_num: int,
                conv_scaler: int,
                dropout: Optional[float] = None):
        return build_network(network_type='fnet',
                             classification=False,
                             in_channel=1,
                             out_channel=1,
                             img_size=image_size,
                             dropout=dropout,
                             num_conv_layers=layer_num,
                             conv_scaler=conv_scaler,
                             layer_components='3gcl',
                             num_group=8,
                             prediction=False)

    def test_unet3d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.unet_3d(image_size=i,
                                  layer_num=5,
                                  conv_scaler=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet2d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.unet_2d(image_size=i,
                                  layer_num=5,
                                  conv_scaler=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_resunet_3d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.resunet_3d(image_size=i,
                                     layer_num=5,
                                     conv_scaler=j,
                                     dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_resunet_2d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.resunet_2d(image_size=i,
                                     layer_num=5,
                                     conv_scaler=j,
                                     dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet3plus_2d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.unet3plus_2d(image_size=i,
                                       layer_num=5,
                                       conv_scaler=j,
                                       dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet3plus_3d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.unet3plus_3d(image_size=i,
                                       layer_num=5,
                                       conv_scaler=j,
                                       dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_fnet_2d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i))

                nn = self.fnet_2d(image_size=i,
                                  layer_num=5,
                                  conv_scaler=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_fnet_3d(self):
        for i in self.image_sizes:
            for j in self.conv_scaler:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.fnet_3d(image_size=i,
                                  layer_num=5,
                                  conv_scaler=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)
