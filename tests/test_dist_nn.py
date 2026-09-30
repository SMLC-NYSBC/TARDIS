#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################

import torch

from tardis_em.dist_pytorch.dist import DIST
from tardis_em.dist_pytorch.model.layers import DistLayer, DistStack


def rand_tensor(shape: tuple):
    return torch.rand(shape)


def randomize(model):
    # Several output layers start at zero, which would hide most of the model
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)


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

    def test_dist_no_grad_matches_autograd(self):
        # Without autograd the layers run a chunked, in-place path; it must
        # give the same result as the regular forward pass
        for structure in ["triang", "dualtriang", "full"]:
            for node_input, node_dim in [(0, None), (3, 16)]:
                torch.manual_seed(0)
                model = DIST(
                    n_out=1,
                    node_input=node_input,
                    node_dim=node_dim,
                    edge_dim=16,
                    num_layers=2,
                    num_heads=4,
                    coord_embed_sigma=[1.0, 10.0, 8],
                    structure=structure,
                    predict=True,
                )
                randomize(model)
                model.eval()

                coords = rand_tensor((2, 20, 3)) * 10
                node = rand_tensor((2, 20, 3)) if node_input > 0 else None

                expected = model(coords=coords, node_features=node).detach()

                default_chunk = DistLayer.inference_chunk_size
                DistLayer.inference_chunk_size = 150  # force several row blocks
                try:
                    with torch.no_grad():
                        x = model(coords=coords, node_features=node)
                finally:
                    DistLayer.inference_chunk_size = default_chunk

                assert torch.allclose(x, expected, rtol=1e-5, atol=1e-6)

    def test_dist_stack_keeps_input(self):
        stack = DistStack(pairs_dim=16, num_layers=2, num_heads=4, structure="triang")
        randomize(stack)
        edge = rand_tensor((1, 10, 10, 16))
        edge_copy = edge.clone()

        with torch.no_grad():
            _, x = stack(edge_features=edge)

        assert torch.equal(edge, edge_copy)
        assert not torch.equal(x, edge)
