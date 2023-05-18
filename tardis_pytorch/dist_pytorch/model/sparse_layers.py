#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

import torch
import torch.nn as nn


def sparse_sigmoid(coo_tensor: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Input: sparse_coo_tensor[Batch x Length x Length x Channel]
    Output: sparse_coo_tensor[Batch x Length x Length x Channel]
    """

    return coo_tensor._values().sigmoid_()


class SparseNorm(nn.Module):
    """
    Input: sparse_coo_tensor[Batch x Length x Length x Channel]
    Output: sparse_coo_tensor[Batch x Length x Length x Channel]
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        g_shape = x.shape

        return torch.sparse_coo_tensor(
            indices=x._indices(),
            values=self.layer_norm(x._values()),
            size=(g_shape[0], g_shape[1], g_shape[2], g_shape[3]),
        )


class SparseLinear(nn.Module):
    """
    Support for batch = 1!!
    Input: sparse_coo_tensor[Batch x Length x Length x Channel]
    Output: sparse_coo_tensor[Batch x Length x Length x Channel]
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        g_shape = x.shape

        return torch.sparse_coo_tensor(
            indices=x._indices(),
            values=self.linear(x._values()),
            size=(g_shape[0], g_shape[1], g_shape[2], self.out_features),
        )


class SparsTriangularUpdate(nn.Module):
    def __init__(self, input_dim: int, channel_dim=128, axis=1):
        super(SparsTriangularUpdate).__init__()

        self.axis = axis
        self.init_scaling = 1 / 1.4142135623730951

        self.norm_input = SparseNorm(input_dim)

        self.linear_a = SparseLinear(input_dim, channel_dim)
        self.gate_a = SparseLinear(input_dim, channel_dim)

        self.linear_b = SparseLinear(input_dim, channel_dim)
        self.gate_b = SparseLinear(input_dim, channel_dim)

        self.norm_o = SparseNorm()
        self.gate_o = SparseLinear(input_dim, input_dim)
        self.linear_o = SparseLinear(channel_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initial parameter and bias scaling.
        """
        nn.init.xavier_uniform_(self.linear_a.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_a.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_b.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_b.bias, 0.0)

        nn.init.constant_(self.linear_o.weight, 0.0)
        nn.init.constant_(self.linear_o.bias, 0.0)

    def forward(self, z: torch.sparse_coo_tensor, k: int) -> torch.sparse_coo_tensor:
        z_shape = z.shape
        z_value_shape = z._values().shape

        z = self.norm_input(z)

        a = torch.sigmoid(self.gate_a(z)) * self.linear_a(z)  # B x nzz x O
        a = a.reshape((z_value_shape[0]//k, k, z_shape[3]))  # B x L x K x O
        b = torch.sigmoid(self.gate_b(z)) * self.linear_b(z)  # B x nzz x O
        b = b.reshape((z_value_shape[0] // k, k, z_shape[3]))  # B x L x K x O

        # i,j -> i,k j,k
        if self.axis == 1:
            k = torch.sparse_coo_tensor(
                indices=z._indices(),
                values=torch.einsum("iko,jko->iko", a, b).reshape((z_value_shape[0], z_shape[3])),
                size=(z_shape[0], z_shape[1], z_shape[2], z_shape[3]),
            )
        else:
            k = torch.sparse_coo_tensor(
                indices=z._indices(),
                values=torch.einsum("kio,kjo->iko", a, b).reshape((z_value_shape[0], z_shape[3])),
                size=(z_shape[0], z_shape[1], z_shape[2], z_shape[3]),
            )

        return torch.sigmoid(self.gate_o(z)) * self.linear_o(self.norm_o(k))
