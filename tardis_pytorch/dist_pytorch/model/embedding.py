#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeEmbedding(nn.Module):
    """
    NODE FEATURE EMBEDDING

    Input: Batch x Length x Dim or Batch x Length
    Output: Batch x Length x Dim

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
    """

    def __init__(self, n_in: int, n_out: int, sigma=1):
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
        Forward node feature embedding.

        Input: Batch x Length x Dim
        Output: Batch x Length x Dim

        Args:
            input_node (torch.Tensor): Node features (RGB or image patches).

        Returns:
            torch.Tensor: Embedded features.
        """
        if input_node is None:
            return None

        if self.linear is not None:
            return self.linear(input_node)
        return torch.cos(F.linear(input_node, self.weight / self.sigma, self.bias))


class EdgeEmbedding(nn.Module):
    """
    COORDINATE EMBEDDING INTO GRAPH

    Set of coordinates is used to build distance matrix which is then
    normalized using negative parabolic function.

    Input: Batch x Length x Dim
    Output: Batch x Length x Length x Dim

    TODO: Buckets
    TODO: Cos expansion

    Args:
        n_out (int): Number of features to output.
        sigma (int, optional tuple): Sigma value for an exponential function is
            used to normalize distances.
    """

    def __init__(self, n_out: int, sigma: Union[int, float, list]):
        super().__init__()
        if isinstance(sigma, list):
            self._range = torch.arange(sigma[0], sigma[1], sigma[2])  # torch.linspace
            assert (
                len(self._range) <= n_out
            ), f"Sigma range is out of shape. n_out = {n_out} but sigma range = {len(self._range)}"
            if len(self._range) == n_out:
                self.linear = None
            else:
                self.linear = nn.Linear(len(sigma), n_out, bias=False)
        else:
            self.linear = nn.Linear(1, n_out, bias=False)
        self.n_out = n_out
        self.sigma = sigma

    def forward(self, input_coord: torch.Tensor) -> torch.Tensor:
        """
        Forward node feature embedding.

        Args:
            input_coord (torch.Tensor): Edge features ([N, 2] or [N, 3]
                coordinates array).

        Returns:
            torch.Tensor: Embedded features.
        """
        g_len = input_coord.shape[1]
        g_range = range(g_len)

        dist = torch.cdist(input_coord, input_coord)
        if isinstance(self.sigma, (int, float)):
            dist = torch.exp(-(dist**2) / (self.sigma**2 * 2))
            isnan = torch.isnan(dist)
            dist = torch.where(isnan, torch.zeros_like(dist), dist)

            # Overwrite diagonal with 1
            dist[:, g_range, g_range] = 1
            return self.linear(dist.unsqueeze(3))
        else:
            dist_range = torch.zeros(
                (1, g_len, g_len, len(self._range)), device=dist.device
            )
            for id, i in enumerate(self._range):
                dist_range[:, :, :, id] = torch.exp(-(dist**2) / (i**2 * 2))

            isnan = torch.isnan(dist_range)
            dist_range = torch.where(isnan, torch.zeros_like(dist_range), dist_range)
            dist_range[:, g_range, g_range, :] = 1

            if self.linear is not None:
                return self.linear(dist_range)
            return dist_range  # Log space