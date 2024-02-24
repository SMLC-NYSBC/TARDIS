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
    NODE FEATURE EMBEDDING

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
            node = self.linear(input_node)
        else:
            node = torch.cos(F.linear(input_node, self.weight / self.sigma, self.bias))

        return (node + 1) / 2


class EdgeEmbedding(nn.Module):
    """
    COORDINATE EMBEDDING INTO GRAPH

    Set of coordinates is used to build distance matrix which is then
    normalized using negative parabolic function.

    Input: Batch x Length x Dim
    Output: Batch x Length x Length x Dim

    Args:
        n_out (int): Number of features to output.
        sigma (int, optional tuple): Sigma value for an exponential function is
            used to normalize distances.
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
        Forward node feature embedding.

        Args:
            input_coord (torch.Tensor): Edge features ([N, 2] or [N, 3]
                coordinates array).

        Returns:
            torch.Tensor: Embedded features.
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
