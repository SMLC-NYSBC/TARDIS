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
from math import sqrt, pi
import time
import numpy as np
from itertools import compress


def gelu(x: list) -> list:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function to a sparse tensor.

    This function applies the GELU activation function element-wise to the values of
    a sparse tensor while preserving its sparsity.

    Args:
        x (torch.sparse_coo_tensor): A sparse coordinate tensor to which the GELU function will be applied.

    Returns:
        torch.sparse_coo_tensor: A new sparse coordinate tensor with the GELU function applied to its values.
    """

    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2)))


class GeluFeedForward(nn.Module):
    """
    Module for a sparse feedforward NN with GELU activation and sparse operations.

    This class defines a two-layer feedforward neural network that operates on sparse data.
    The network includes normalization, linear transformations, and a GELU activation function.
    """

    def __init__(self, input_dim: int, ff_dim: int):
        """
        Initializes the SparseGeluFeedForward.

        Args:
            input_dim (int): The dimensionality of the input data.
            ff_dim (int): The dimensionality of the output from the first linear transformation.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)

        # Initialize the weights and biases of the second linear transformation to zero
        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: list) -> list:
        """
        Forward pass for SparseGeluFeedForward.

        Args:
            x (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing the output from the feedforward network.
        """
        x = self.norm(x)  # Apply normalization
        x = self.linear1(x)  # Apply first linear transformation
        x = self.linear2(
            gelu(x)
        )  # Apply GELU activation and second linear transformation

        return x


class SparsTriangularUpdate(nn.Module):
    """
    Module for updating a sparse tensor in a triangular fashion.

    This class applies a sequence of transformations, including normalization,
    sigmoid activation, linear transformations, and a triangular multiplication update.
    The transformations are designed to update the input tensor while preserving its sparsity.
    """

    def __init__(self, input_dim: int, channel_dim=128, axis=1, k=12):
        """
        Initializes the SparsTriangularUpdate.

        Args:
            input_dim (int): The dimensionality of the input data.
            channel_dim (int): The number of channels in the output from some transformations.
            axis (int): The axis to be used in the triangular update rule.
            k (int): The number of top elements to be used in the triangular update rule.
        """
        super().__init__()
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.k = k
        self.init_scaling = 1 / sqrt(2)
        # self.init_scaling = sqrt(2 / (1 + pi / 2))

        # Define the transformations to be applied
        self.norm_input = nn.LayerNorm(input_dim)

        self.linear_a = nn.Linear(input_dim, channel_dim)
        self.gate_a = nn.Linear(input_dim, channel_dim)

        self.linear_b = nn.Linear(input_dim, channel_dim)
        self.gate_b = nn.Linear(input_dim, channel_dim)

        self.norm_o = nn.LayerNorm(channel_dim)
        self.gate_o = nn.Linear(input_dim, input_dim)
        self.linear_o = nn.Linear(channel_dim, input_dim)

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

    def forward(self, x: torch.tensor, idx: list) -> list:
        """
        Forward pass for SparsTriangularUpdate.

        Args:
            x (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing the updated tensor.
        """
        x_value_shape = x.shape

        x = self.norm_input(x)  # Batch x Length x Channels [N, Ch]

        # Compute intermediate transformations
        # [B, M, KNN, Ch]

        a = torch.sigmoid(self.gate_a(x)) * self.linear_a(x)
        b = torch.sigmoid(self.gate_b(x)) * self.linear_b(x)

        # # Apply triangular multiplication update
        k = torch.zeros_like(a)

        if self.axis == 1:  # Row-wise update
            for i in range(len(idx[1][1])):
                i_element = idx[1][1][i].copy()
                i_element.append(i)

                for _id, j in enumerate(idx[1][1][i]):
                    fil = [True if m in i_element else False for m in idx[1][1][j]]
                    j_th = list(compress(idx[1][1][j], fil))

                    fil = [True if m in idx[1][1][j] else False for m in i_element]
                    i_th = list(compress(i_element, fil))

                    k[:, idx[1][0][i][_id], :] = torch.sum(
                        a[:, i_th, :] * b[:, j_th, :], dim=1
                    )
        else:
            for i in range(len(idx[2][1])):
                i_element = idx[2][1][i].copy()
                i_element.append(i)

                for _id, j in enumerate(idx[2][1][i]):
                    fil = [True if m in i_element else False for m in idx[2][1][j]]
                    j_th = list(compress(idx[2][1][j], fil))

                    fil = [True if m in idx[2][1][j] else False for m in i_element]
                    i_th = list(compress(i_element, fil))

                    k[:, idx[2][0][i][_id], :] = torch.sum(
                        a[:, i_th, :] * b[:, j_th, :], dim=1
                    )

        # if self.axis == 1:  # Row-wise
        #     idx[0] = idx[0].reshape(org_shape[1], self.k, 2)
        #     a = a.reshape(1, org_shape[1], self.k, ch)
        #     idx[0] = idx[0].reshape(org_shape[1], self.k, 2)
        #     b = b.reshape(1, org_shape[1], self.k, ch)

        #     k = a.repeat_interleave(self.k, dim=1) * b[0, idx[0][..., 1].flatten().long(), :].unsqueeze(0)
        #     k = torch.sum(k, dim=2)
        #     k = k.reshape(1, M_len, 32)

        #     idx = [idx[0].reshape(org_shape[1] * self.k, 2), idx[1]]
        # else:  # Column-wise
        #     for _id in range(mm_len):
        #         k[0, i_id, :] = sparse_mm(a[1][:, :mm_len, :][:, i_id, :],
        #                                        b[1][:, :mm_len, :][:, i_id, :],
        #                                        a[0][:mm_len, :][i_id, :])

        return torch.sigmoid(self.gate_o(x)) * self.linear_o(self.norm_o(k)), idx


def sparse_to_dense(x: list, numpy=False):
    _shape = x[-1]

    if numpy:
        try:
            idx = x[0].cpu().detach().numpy()
            x = x[1].cpu().detach().numpy()[0, ...]
        except:
            idx = x[0].detach().numpy()
            x = x[1].detach().numpy()[0, ...]

        graph = np.zeros(_shape, dtype=np.float16)
    else:
        idx = x[0]
        x = x[1][0, ...]
        graph = torch.zeros(_shape, dtype=torch.float32, device=x.device)

    for _id, i in enumerate(idx):
        i, j = i
        graph[0, i, j, ...] = x[_id, ...]

    graph[0, range(_shape[1]), range(_shape[2]), ...] = 1

    return graph
