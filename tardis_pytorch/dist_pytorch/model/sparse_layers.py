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
import math


def sparse_sigmoid(coo_tensor: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Input: sparse_coo_tensor[Batch x Length x Length x Channel]
    Output: sparse_coo_tensor[Batch x Length x Length x Channel]
    """
    coo_tensor._values().sigmoid_()
    return coo_tensor


class SparseNorm(nn.Module):
    """
    Input: sparse_coo_tensor[Batch x Length x Length x Channel]
    Output: sparse_coo_tensor[Batch x Length x Length x Channel]
    """

    def __init__(self, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        with torch.no_grad():
            non_zero_values = x._values()
            mean = non_zero_values.mean()
            variance = non_zero_values.var(unbiased=False)
            normalized_values = (non_zero_values - mean) / torch.sqrt(
                variance + self.eps
            )

        return torch.sparse_coo_tensor(
            x._indices(), self.alpha * normalized_values + self.beta, x.shape
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
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        B, C, R = x._indices()
        g_shape = x.shape

        output_values = [
            torch.matmul(self.weight, x[0, ci, ri])
            if self.bias
            else torch.matmul(self.weight, x[0, ci, ri]) + self.bias
            for ci, ri in zip(C, R)
        ]
        # for c, r in zip(C, R):
        #     transformed_value = torch.matmul(self.weight, x[0, c, r])
        #     if self.bias is not None:
        #         transformed_value = transformed_value + self.bias
        #
        #     output_values.append(transformed_value)

        return torch.sparse_coo_tensor(
            indices=x._indices(),
            values=torch.stack(output_values),
            size=(g_shape[0], g_shape[1], self.out_features),
        )


class SparsTriangularUpdate(nn.Module):
    def __init__(self, input_dim: int, channel_dim=128, axis=1):
        super(SparsTriangularUpdate).__init__()

        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.init_scaling = 1 / 1.4142135623730951

        self.norm_input = SparseNorm()

        self.linear_a = SparseLinear(input_dim, channel_dim)
        self.gate_a = SparseLinear(input_dim, channel_dim)

        self.linear_b = SparseLinear(input_dim, channel_dim)
        self.gate_b = SparseLinear(input_dim, channel_dim)

        self.norm_o = SparseNorm()
        self.gate_o = SparseLinear(input_dim, input_dim)
        self.linear_o = SparseLinear(channel_dim, input_dim)

        self._reset_parameters()

    @staticmethod
    def sparse_batched_dot_product(
        a: torch.sparse_coo_tensor, b: torch.sparse_coo_tensor
    ) -> torch.sparse_coo_tensor:
        # Assuming a and b are 4D tensors of shape [batch_size, i, j, o]
        batch_size, i_dim, _, o_dim = a.shape
        result = torch.zeros((batch_size, i_dim, i_dim, o_dim), device=a.device)

        for b_i in range(batch_size):
            for i in range(i_dim):
                result[b_i, i] = a[b_i, i] @ b[b_i]

        return result

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

    def forward(self, z: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        z = self.norm_input(z)

        a = torch.sigmoid(self.gate_a(z)) * self.linear_a(z)  # B x L x L x O
        b = torch.sigmoid(self.gate_b(z)) * self.linear_b(z)  # B x L x L x O

        # i,j -> i,k j,k
        if self.axis == 1:
            k = self.sparse_batched_dot_product(a, b)
        else:
            # Transpose a and b for the second axis
            a = a.permute(0, 2, 1, 3)
            b = b.permute(0, 2, 1, 3)
            k = self.sparse_batched_dot_product(a, b)

        return torch.sigmoid(self.gate_o(z)) * self.linear_o(self.norm_o(k))
