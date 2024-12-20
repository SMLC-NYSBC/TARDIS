#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import torch

from tardis_em.dist_pytorch.dist import DIST


def rand_tensor(shape: tuple):
    return torch.rand(shape)


class TestGraphFormer:
    def test_dist_wo_rgb(self):
        for n_dim in [32, 16, None]:
            for e_dim in [32, 16]:
                for n_layer in [3, 1]:
                    for n_head in [4, 4, 1]:
                        model = DIST(
                            n_out=1,
                            node_input=0,
                            node_dim=n_dim,
                            edge_dim=e_dim,
                            num_layers=n_layer,
                            num_heads=n_head,
                            num_cls=None,
                            dropout_rate=0,
                            coord_embed_sigma=16,
                            predict=False,
                        )
                        x = model(coords=rand_tensor((1, 5, 3)), node_features=None)
                        assert x.shape == torch.Size((1, 1, 5, 5))

                        x = model(coords=rand_tensor((1, 5, 2)), node_features=None)
                        assert x.shape == torch.Size((1, 1, 5, 5))

    def test_dist_w_rgb(self):
        for n_dim in [32, 16]:
            for e_dim in [32, 16]:
                for n_layer in [3, 1]:
                    for n_head in [4, 4, 1]:
                        model = DIST(
                            n_out=1,
                            node_input=3,
                            node_dim=n_dim,
                            edge_dim=e_dim,
                            num_layers=n_layer,
                            num_heads=n_head,
                            dropout_rate=0,
                            coord_embed_sigma=16,
                            predict=False,
                        )
                        x = model(
                            coords=rand_tensor((1, 5, 3)),
                            node_features=rand_tensor((1, 5, 3)),
                        )
                        assert x.shape == torch.Size((1, 1, 5, 5))

                        model = DIST(
                            n_out=1,
                            node_input=3,
                            node_dim=n_dim,
                            edge_dim=e_dim,
                            num_layers=n_layer,
                            num_heads=n_head,
                            dropout_rate=0,
                            coord_embed_sigma=16,
                            predict=False,
                        )
                        x = model(
                            coords=rand_tensor((1, 5, 2)),
                            node_features=rand_tensor((1, 5, 3)),
                        )
                        assert x.shape == torch.Size((1, 1, 5, 5))
