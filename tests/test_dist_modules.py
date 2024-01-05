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

from tardis_em.dist_pytorch.model.modules import (
    ComparisonLayer,
    gelu,
    GeluFeedForward,
    PairBiasSelfAttention,
    QuadraticEdgeUpdate,
    SelfAttention2D,
    TriangularEdgeUpdate,
)


def test_comparison_layer():
    data = torch.rand((10, 1, 32))  # (Length x Batch x Channels)
    compare = ComparisonLayer(input_dim=32, output_dim=64, channel_dim=64)
    with torch.no_grad():
        data_compare = compare(data)
    assert data_compare.shape == torch.Size((1, 10, 10, 64))


def test_gelu_forward():
    g_forward = GeluFeedForward(input_dim=5, ff_dim=2)
    data = torch.rand((1, 10, 10, 5))

    with torch.no_grad():
        data_gelu = g_forward(data)

    assert data.shape == data_gelu.shape
    assert torch.all(data != data_gelu)  # Check if data are modified


def test_pair_attention():
    data_q = torch.rand((10, 1, 32))  # (Length x Batch x Channels)
    data_p = torch.rand((1, 10, 10, 64))  # (Batch x Length x Length x Channels)
    pair_attn = PairBiasSelfAttention(embed_dim=32, pairs_dim=64, num_heads=8)

    with torch.no_grad():
        data_attn = pair_attn(query=data_q, pairs=data_p)

    assert data_attn.shape == data_q.shape


def test_quadratic_attn():
    data = torch.rand((1, 10, 10, 64))
    quad_0 = QuadraticEdgeUpdate(input_dim=64, axis=0)
    quad_1 = QuadraticEdgeUpdate(input_dim=64)

    with torch.no_grad():
        q_0 = quad_0(data)
        q_1 = quad_1(data)

    assert q_0.shape == q_1.shape
    assert torch.all(q_0 == q_1)


def test_triang_attn():
    data = torch.rand((1, 10, 10, 64))
    quad_0 = TriangularEdgeUpdate(input_dim=64, axis=0)
    quad_1 = TriangularEdgeUpdate(input_dim=64)

    with torch.no_grad():
        q_0 = quad_0(data)
        q_1 = quad_1(data)

    assert q_0.shape == q_1.shape
    assert torch.all(q_0 == q_1)


def test_self_attn():
    data = torch.rand((1, 10, 10, 32))
    self_attn_0 = SelfAttention2D(embed_dim=32, num_heads=8, axis=0)
    self_attn_1 = SelfAttention2D(embed_dim=32, num_heads=8, axis=1)

    with torch.no_grad():
        data_attn_0 = self_attn_0(data)
        data_attn_1 = self_attn_1(data)

    assert data_attn_0.shape == data_attn_1.shape
    assert torch.all(data_attn_0 == data_attn_1)


def test_gelu():
    data = torch.rand((1, 10, 10))

    with torch.no_grad():
        data_gelu = gelu(data)
    assert data.shape == data_gelu.shape
    assert torch.all(data != data_gelu)  # Check if data are modified
