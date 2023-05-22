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
from math import sqrt


def sparse_operation(x, y, z=None, op="sum") -> torch.sparse_coo_tensor:
    """
    Perform sparse operations on tensors.

    Args:
        x (torch.sparse_coo_tensor): First input tensor.
        y (torch.sparse_coo_tensor): Second input tensor.
        z (torch.sparse_coo_tensor, optional): Third input tensor. Defaults to None.
        op (str, optional): Operation to perform. Can be 'sum', 'subtract', 'divide', or 'multiply'. Defaults to 'sum'.

    Returns:
        torch.sparse_coo_tensor: Resulting sparse tensor.
    """
    assert op in ["sum", "subtract", "divide", "multiply"]

    g_shape = x.shape

    if z is not None:
        assert op in ["sum", "subtract"]
        if op == "sum":
            # Perform element-wise addition of values from x, y, and z tensors
            result_values = x._values() + y._values() + z._values()
        else:
            # Perform element-wise subtraction of values from x, y, and z tensors
            result_values = x._values() - y._values() - z._values()
    else:
        if op == "sum":
            # Perform element-wise addition of values from x and y tensors
            result_values = x._values() + y._values()
        elif op == "subtract":
            # Perform element-wise subtraction of values from x and y tensors
            result_values = x._values() - y._values()
        elif op == "divide":
            # Perform element-wise division of values from x and y tensors
            result_values = x._values() / y._values()
        else:
            # Perform element-wise multiplication of values from x and y tensors
            result_values = x._values() * y._values()

    # Create a sparse tensor with the computed values and the same indices and size as x tensor
    return torch.sparse_coo_tensor(
        indices=x._indices(),
        values=result_values,
        size=(g_shape[0], g_shape[1], g_shape[2], g_shape[3]),
    )


def sparse_sigmoid(x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Applies the sigmoid function to a sparse tensor.

    This function applies the sigmoid function element-wise to the values of
    a sparse tensor while preserving its sparsity.

    Args:
        x (torch.sparse_coo_tensor): A sparse coordinate tensor to which the sigmoid function will be applied.

    Returns:
        torch.sparse_coo_tensor: A new sparse coordinate tensor with the sigmoid function applied to its values.
    """
    g_shape = x.shape

    return torch.sparse_coo_tensor(
        indices=x._indices(),
        values=torch.sigmoid(x._values()),
        size=(g_shape[0], g_shape[1], g_shape[2], g_shape[3]),
    )


def sparse_gelu(x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function to a sparse tensor.

    This function applies the GELU activation function element-wise to the values of
    a sparse tensor while preserving its sparsity.

    Args:
        x (torch.sparse_coo_tensor): A sparse coordinate tensor to which the GELU function will be applied.

    Returns:
        torch.sparse_coo_tensor: A new sparse coordinate tensor with the GELU function applied to its values.
    """
    g_shape = x.shape

    return torch.sparse_coo_tensor(
        indices=x._indices(),
        values=x._values() * 0.5 * (1.0 + torch.erf(x._values() / sqrt(2))),
        size=(g_shape[0], g_shape[1], g_shape[2], g_shape[3]),
    )


class SparseGeluFeedForward(nn.Module):
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
        self.norm = SparseNorm(input_dim)
        self.linear1 = SparseLinear(input_dim, ff_dim)
        self.linear2 = SparseLinear(ff_dim, input_dim)

        # Initialize the weights and biases of the second linear transformation to zero
        nn.init.constant_(self.linear2.linear.weight, 0.0)
        nn.init.constant_(self.linear2.linear.bias, 0.0)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
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
            sparse_gelu(x)
        )  # Apply GELU activation and second linear transformation

        return x


class SparseNorm(nn.Module):
    """
    Module for applying layer normalization to sparse tensors.

    This class applies the Layer Normalization operation to a sparse tensor,
    preserving its sparsity.
    """

    def __init__(self, input_dim: int):
        """
        Initializes the SparseNorm.

        Args:
            input_dim (int): The dimensionality of the input data.
        """
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass for SparseNorm.

        Args:
            x (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor with layer normalization applied to its values.
        """
        g_shape = x.shape

        return torch.sparse_coo_tensor(
            indices=x._indices(),
            values=self.layer_norm(x._values()),
            size=(g_shape[0], g_shape[1], g_shape[2], g_shape[3]),
        )


class SparseLinear(nn.Module):
    """
    Module for applying a linear transformation to sparse tensors.

    This class applies a linear transformation to a sparse tensor, preserving its sparsity.
    The input and output tensors have the shape [Batch x Length x Length x Channel].
    Currently, this class supports a batch size of 1.
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        Initializes the SparseLinear.

        Args:
            in_features (int): The number of input features (i.e., input tensor's Channel dimension).
            out_features (int): The number of output features (i.e., output tensor's Channel dimension).
            bias (bool): A boolean value indicating whether to include a bias term in the linear transformation.
        """
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass for SparseLinear.

        Args:
            x (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor with a linear transformation applied to its values.
        """
        g_shape = x.shape

        return torch.sparse_coo_tensor(
            indices=x._indices(),
            values=self.linear(x._values()),
            size=(g_shape[0], g_shape[1], g_shape[2], self.out_features),
        )


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

        # Define the transformations to be applied
        self.norm_input = SparseNorm(input_dim)

        self.linear_a = SparseLinear(input_dim, channel_dim)
        self.gate_a = SparseLinear(input_dim, channel_dim)

        self.linear_b = SparseLinear(input_dim, channel_dim)
        self.gate_b = SparseLinear(input_dim, channel_dim)

        self.norm_o = SparseNorm(channel_dim)
        self.gate_o = SparseLinear(input_dim, input_dim)
        self.linear_o = SparseLinear(channel_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initial parameter and bias scaling.
        """
        nn.init.xavier_uniform_(self.linear_a.linear.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_a.linear.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_b.linear.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_b.linear.bias, 0.0)

        nn.init.constant_(self.linear_o.linear.weight, 0.0)
        nn.init.constant_(self.linear_o.linear.bias, 0.0)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass for SparsTriangularUpdate.

        Args:
            x (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing the updated tensor.
        """
        x_shape = x.shape
        x_value_shape = x._values().shape

        x = self.norm_input(x)

        # Compute intermediate transformations
        a = sparse_operation(
            sparse_sigmoid(self.gate_a(x)), self.linear_a(x), op="multiply"
        )
        a = a._values().reshape((x_value_shape[0] // self.k, self.k, self.channel_dim))

        b = sparse_operation(
            sparse_sigmoid(self.gate_b(x)), self.linear_b(x), op="multiply"
        )
        b = b._values().reshape((x_value_shape[0] // self.k, self.k, self.channel_dim))

        # Apply triangular multiplication update
        if self.axis == 1:
            k = torch.sparse_coo_tensor(
                indices=x._indices(),
                values=torch.einsum("iko,jko->iko", a, b).reshape(
                    (x_value_shape[0], self.channel_dim)
                ),
                size=(x_shape[0], x_shape[1], x_shape[2], self.channel_dim),
            )
        else:
            k = torch.sparse_coo_tensor(
                indices=x._indices(),
                values=torch.einsum("kio,kjo->iko", a, b).reshape(
                    (x_value_shape[0], self.channel_dim)
                ),
                size=(x_shape[0], x_shape[1], x_shape[2], self.channel_dim),
            )

        return sparse_operation(
            sparse_sigmoid(self.gate_o(x)), self.linear_o(self.norm_o(k)), op="multiply"
        )
