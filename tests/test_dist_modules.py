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


def test_self_attn_axes():
    torch.manual_seed(0)
    data = torch.rand((2, 6, 7, 16))  # (Batch x Rows x Cols x Channels)

    for axis in [0, 1, None]:
        attn = SelfAttention2D(embed_dim=16, num_heads=4, axis=axis)
        torch.nn.init.normal_(attn.out_proj.weight, std=0.1)  # zero at init

        changed = data.clone()
        changed[0, 2, 3] += 1.0
        with torch.no_grad():
            diff = (attn(changed) - attn(data)).abs().sum(-1) > 1e-6

        # Only positions sharing the attended row, column, or grid see the change
        expected = torch.zeros_like(diff)
        if axis == 0:
            expected[0, :, 3] = True
        elif axis == 1:
            expected[0, 2, :] = True
        else:
            expected[0] = True
        assert torch.equal(diff, expected)

        # Splitting into small attention batches gives the same result
        attn_split = SelfAttention2D(embed_dim=16, num_heads=4, axis=axis, max_size=100)
        attn_split.load_state_dict(attn.state_dict())
        with torch.no_grad():
            assert torch.allclose(attn_split(data), attn(data), atol=1e-6)


def test_self_attn_padding_mask():
    torch.manual_seed(0)
    data = torch.rand((1, 5, 5, 16))
    mask = torch.zeros((1, 5, 5), dtype=torch.bool)
    mask[0, :, 4] = True  # ignore column 4 as keys

    attn = SelfAttention2D(embed_dim=16, num_heads=4, axis=1)
    torch.nn.init.normal_(attn.out_proj.weight, std=0.1)

    changed = data.clone()
    changed[0, :, 4] += 1.0
    with torch.no_grad():
        out = attn(data)
        out_changed = attn(changed)
        out_masked = attn(data, padding_mask=mask)
        out_masked_changed = attn(changed, padding_mask=mask)

    # Without the mask, column 4 affects the other columns (attention over j);
    # with the mask, the masked keys do not
    assert not torch.allclose(out[:, :, :4], out_changed[:, :, :4], atol=1e-6)
    assert torch.allclose(out_masked[:, :, :4], out_masked_changed[:, :, :4], atol=1e-6)


def test_gelu():
    data = torch.rand((1, 10, 10))

    with torch.no_grad():
        data_gelu = gelu(data)
    assert data.shape == data_gelu.shape
    assert torch.all(data != data_gelu)  # Check if data are modified
