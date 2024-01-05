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

from tardis_em.dist_pytorch.model.embedding import EdgeEmbedding, NodeEmbedding


def test_node_embedding():
    test_data = torch.Tensor(([0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.1, 0.1, 0.1]))
    test_data = test_data[None, :]
    assert test_data.shape == torch.Size([1, 3, 3])
    node_embed = NodeEmbedding(n_in=3, n_out=12)

    node = node_embed(test_data)
    assert node.shape == torch.Size([1, 3, 12])

    test_data = torch.Tensor(([0.1], [0.2], [0.1]))
    test_data = test_data[None, :]
    assert test_data.shape == torch.Size([1, 3, 1])
    node_embed = NodeEmbedding(n_in=1, n_out=12)

    node = node_embed(test_data)
    assert node.shape == torch.Size([1, 3, 12])


def test_edge_embedding():
    test_data = torch.Tensor(
        (
            [1.0028e-01, 1.6779e00, 1.3603e00],
            [-1.3039e-02, 1.6830e00, 2.0412e-01],
            [3.6169e-01, 1.5056e00, 1.9426e-01],
            [9.5721e-01, 1.4940e00, 1.2675e-01],
            [6.0100e-01, 1.4349e00, 5.5771e-01],
            [3.0420e-01, 1.5386e00, 9.7365e-01],
            [1.7228e-01, 1.5315e00, 1.7924e-01],
            [7.5742e-01, 1.4056e00, 8.6392e-01],
            [6.4784e-01, 1.4155e00, 4.5924e-01],
            [1.6119e-01, 1.5376e00, 2.3625e-01],
            [8.5796e-01, 1.3776e00, 1.4127e-01],
            [8.4867e-01, 1.6888e00, 1.3564e-01],
            [1.0595e00, 1.3240e00, 8.6453e-01],
            [1.1620e-01, 1.7021e00, 1.5622e00],
            [1.5741e-01, 1.5533e00, 3.6822e-01],
            [6.6234e-01, 1.6897e00, 1.5364e-01],
            [5.9537e-01, 1.4365e00, 4.7205e-01],
            [5.3270e-01, 1.4525e00, 2.6412e-01],
            [1.1602e00, 1.3018e00, 1.1159e-01],
            [6.9506e-01, 1.4289e00, 9.7230e-01],
            [6.7114e-01, 1.4119e00, 5.5941e-01],
            [6.2954e-01, 1.4443e00, 8.7431e-01],
            [1.6151e-01, 1.5818e00, 6.6761e-01],
            [6.8229e-01, 1.4276e00, 8.6021e-01],
        )
    )
    test_data = test_data[None, :]
    assert test_data.shape == torch.Size([1, 24, 3])

    edge_embed = EdgeEmbedding(n_out=12, sigma=0.2)
    edge = edge_embed(test_data)
    assert edge.shape == torch.Size([1, 24, 24, 12])
