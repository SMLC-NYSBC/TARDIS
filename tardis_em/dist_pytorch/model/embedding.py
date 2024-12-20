#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """
    NodeEmbedding class for transforming node features using either a learned
    linear mapping or a randomized cosine transformation, depending on the
    value of sigma.

    This class is used to embed input node features (e.g., RGB values or image
    patches) into a desired output dimension. If sigma is 0, a trainable linear
    layer is applied. Otherwise, a fixed random projection is utilized with
    a cosine activation.
    """

    def __init__(self, n_in: int, n_out: int, sigma=1):
        """
        This class initializes a custom layer with weights and biases. Depending on the
        value of sigma, it either initializes a traditional linear layer without a bias
        or computes weights and biases for a custom mathematical operation. It also
        registers these parameters as buffers for later use if sigma is not zero.

        :param n_in: Number of input features.
        :type n_in: int
        :param n_out: Number of output features.
        :type n_out: int
        :param sigma: Standard deviation used for weight initialization. If zero,
            a standard linear layer is initialized instead. Defaults to 1.
        :type sigma: float
        """
        super().__init__()

        self.linear = None
        if sigma == 0:
            self.linear = nn.Linear(n_in, n_out, bias=False)
        else:
            self.n_in = n_in
            self.sigma = torch.tensor(sigma, dtype=torch.float32)

            w = torch.randn(n_out, n_in)
            b = torch.rand(n_out) * 2 * torch.pi

            self.register_buffer("weight", w)
            self.register_buffer("bias", b)

    def forward(
        self, input_node: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Performs the forward pass of the module. If an input tensor is provided, it processes
        it through the defined `linear` layer if available, otherwise applies a specific
        transformation involving the cosine function and linear operation.

        :param input_node: Input tensor to process. If None, returns None.
        :type input_node: Optional[torch.Tensor]
        :return: A tensor where the processed input is transformed and scaled into the
            range [0, 1], or None if no input was provided.
        :rtype: Optional[torch.Tensor]
        """
        if input_node is None:
            return None

        if self.linear is not None:
            node = self.linear(input_node)
        else:
            node = torch.cos(F.linear(input_node, self.weight / self.sigma, self.bias))

        return (node + 1) / 2


class EdgeEmbedding(nn.Module):
    """
    EdgeEmbedding layer encapsulates the functionality of computing edge-based
    representations within a graph, leveraging Gaussian radial basis functions (RBF)
    as embedding mechanisms. This module enables the transformation of edge distances
    into either fixed-dimensional encodings or dynamically adjustable encodings
    based on input configurations.

    EdgeEmbedding computes Gaussian kernel representations of edge distances
    and optionally applies a linear transformation to produce embeddings
    of a specified dimensionality. It supports both fixed sigma values (as single
    or iterable ranges) and learns to dynamically adjust their dimensions
    via linear layers.
    """

    def __init__(self, n_out: int, sigma: Union[int, float, list]):
        super().__init__()

        self.n_out = n_out
        self.sigma = sigma

        if isinstance(sigma, (list, tuple)):
            if len(sigma) == 2:
                self._range = torch.linspace(sigma[0], sigma[1], n_out)
            else:
                self._range = torch.linspace(sigma[0], sigma[1], int(sigma[2]))
            assert (
                len(self._range) <= n_out
            ), f"Sigma range is out of shape. n_out = {n_out} but sigma range = {len(self._range)}"

            if len(self._range) != self.n_out:
                self.linear = nn.Linear(len(self._range), n_out, bias=False)
            else:
                self.linear = None
        else:
            self.linear = nn.Linear(1, n_out, bias=False)

    def forward(self, input_coord: torch.Tensor) -> torch.Tensor:
        """
        Computes a transformation of pairwise distances between input coordinates
            and applies an optional linear transformation. The specific transformation
            depends on the configuration, such as whether a fixed `sigma` value or
            varying `_range` values are provided. Handles missing values robustly by
            replacing NaN distance values with zeros.

        :param input_coord: Tensor of shape (..., L, D) where L is the number of
            coordinate points and D is the dimensionality of each point. It represents
            the input feature coordinates for which pairwise distances will be computed.
        :type input_coord: torch.Tensor

        :return: Tensor of shape (..., L, L, K) where K is either 1 or the length of the `_range` attribute.
            Represents the transformed pairwise distances. If `self.linear` is provided, the returned tensor
            is further transformed using the linear layer.
        :rtype: torch.Tensor
        """
        g_len = input_coord.shape[-2]
        g_range = range(g_len)

        dist = torch.cdist(input_coord, input_coord)
        if isinstance(self.sigma, (int, float)):
            dist = torch.exp(-(dist**2) / (self.sigma**2 * 2))

            dist[torch.isnan(dist)] = 0.0
            # isnan = torch.isnan(dist)
            # dist = torch.where(isnan, torch.zeros_like(dist), dist)

            # Overwrite diagonal with 1
            dist = dist.unsqueeze(3)
            dist[:, g_range, g_range, :] = 1.0
        else:
            _range_expanded = self._range.view(1, 1, 1, -1).to(dist.device)
            dist = dist.unsqueeze(-1)
            dist = torch.exp(-(dist**2) / (_range_expanded**2 * 2))
            dist[torch.isnan(dist)] = 0.0
            dist[:, g_range, g_range, :] = 1.0

            # dist_range = torch.zeros(
            #     (1, g_len, g_len, len(self._range)), device=dist.device
            # )
            #
            # for id_1, i in enumerate(self._range):
            #     dist_range[..., id_1] = torch.exp(-(dist**2) / (i**2 * 2))
            #
            # dist_range[torch.isnan(dist_range)] = 0.0
            # dist_range[:, g_range, g_range, :] = 1.0

        if self.linear is not None:
            return self.linear(dist)
        else:
            return dist
