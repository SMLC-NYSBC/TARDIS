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
    Implements a neural network module for sparsely updating tensors using a triangular update rule.

    The SparsTriangularUpdate class provides a mechanism to apply sparse triangular updates on
    coordinate tensors, enabling efficient computations for specific data structures or patterns.
    The module includes normalization layers, linear transformations, and gating mechanisms
    to modulate and compute updates. It supports customizable parameters for dimensions
    and the axis of operations, as well as a k-nearest-neighbor (knn) connectivity.
    """

    def __init__(self, input_dim: int, channel_dim=128, axis=1, knn=8):
        """
        Class instantiates and configures layers and parameters for a neural network
        module. The layers include LayerNorm and Linear transformations applied to
        inputs based on dimensions specified during initialization. The class
        also ensures proper initialization of parameters to achieve stability.

        :param input_dim: Input feature dimension for the given data.
        :type input_dim: int
        :param channel_dim: Number of output channels for the linear layers. Defaults to 128.
        :type channel_dim: int, optional
        :param axis: Axis along which normalization will be performed. Defaults to 1.
        :type axis: int, optional
        :param knn: Number of nearest neighbors to consider for operations (specific
                    use case dependent). Defaults to 8.
        :type knn: int, optional
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
        Processes input tensor `x` using gated linear transformations and triangular
        multiplication updates with specified row-wise or column-wise operations,
        returning the updated tensor as a result.

        This function applies the following transformations:
        1. Normalizes the input tensor `x`.
        2. Computes intermediate transformations `a` and `b` using sigmoid activations
           and a combination of gate layers (`gate_a`, `gate_b`) along with linear
           layers (`linear_a`, `linear_b`).
        3. Depending on the specified axis, computes either a row-wise or column-wise
           multiplication update using the indices provided.
        4. Applies a final sigmoid activation combined with gated output
           transformation using `gate_o` and `linear_o`.

        :param x: The input tensor. Shape `[N, Ch]` representing
            batch size (`N`) by the number of channels (`Ch`).
        :param indices: List of index tensors used to specify the
            range for row-wise or column-wise updates. For row updates,
            indices[0] is used, and for column updates, indices[1] is used.
        :return: Updated tensor after applying gated transformations
            and triangular multiplication updates. Shape `[N, Ch]`.
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
