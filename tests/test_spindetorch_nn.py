from typing import Optional
import torch
from tardis.spindletorch.utils.build_network import build_network


class TestNetwork3D:
    image_sizes = [32, 64, 96]
    conv_multiplayers = [32, 64]

    def unet(self,
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
                             conv_multiplayer=conv_multiplayer,
                             layer_components='gcl',
                             no_groups=8,
                             prediction=False)

    def resunet(self,
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
                             layer_components='gcl',
                             no_groups=8,
                             prediction=False)

    def unet3plus(self,
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
                             layer_components='gcl',
                             no_groups=8,
                             prediction=False)

    def test_unet(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.unet(image_size=i,
                               layer_no=5,
                               conv_multiplayer=j,
                               dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_resunet(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.resunet(image_size=i,
                                  layer_no=5,
                                  conv_multiplayer=j,
                                  dropout=None)

                with torch.no_grad():
                    input = nn(input)

    def test_unet3plus(self):
        for i in self.image_sizes:
            for j in self.conv_multiplayers:
                # Batch x Channel x Z x Y x X
                input = torch.rand((1, 1, i, i, i))

                nn = self.unet3plus(image_size=i,
                                    layer_no=5,
                                    conv_multiplayer=j,
                                    dropout=None)

                with torch.no_grad():
                    input = nn(input)
