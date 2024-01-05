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
import torch.nn as nn


class SparsTriangularUpdate(nn.Module):
    """
    Module for updating a sparse tensor in a triangular fashion.

    This class applies a sequence of transformations, including normalization,
    sigmoid activation, linear transformations, and a triangular multiplication update.
    The transformations are designed to update the input tensor while preserving its sparsity.
    """

    def __init__(self, input_dim: int, channel_dim=128, axis=1, knn=8):
        """
        Initializes the SparsTriangularUpdate.

        Args:
            input_dim (int): The dimensionality of the input data.
            channel_dim (int): The number of channels in the output from some transformations.
            axis (int): The axis to be used in the triangular update rule.
        """
        super().__init__()
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.init_scaling = 1 / 1.4142135623730951
        self.knn = knn

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

    def forward(self, x: torch.tensor, indices: list) -> torch.tensor:
        """
        Forward pass for SparsTriangularUpdate.

        Args:
            x (torch.tensor): A sparse coordinate tensor containing the input data.
            indices (list): List of all indices

        Returns:
            torch.tensor, list: A sparse coordinate tensor representing the updated tensor.
        """
        x = self.norm_input(x)  # Batch x Length x Channels [N, Ch]

        # Compute intermediate transformations
        a = torch.sigmoid(self.gate_a(x)) * self.linear_a(x)
        b = torch.sigmoid(self.gate_b(x)) * self.linear_b(x)

        # Apply triangular multiplication update

        if self.axis == 1:  # Row-wise update
            k = torch.einsum("ik,ijk->ik", a, b[indices[0]])
        else:  # Col-wise update
            k = torch.einsum("ik,ijk->ik", a, b[indices[1]])

        return torch.sigmoid(self.gate_o(x)) * self.linear_o(self.norm_o(k))
