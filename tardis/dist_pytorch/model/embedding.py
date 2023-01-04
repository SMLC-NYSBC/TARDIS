from typing import Optional, Union

import torch
import torch.nn as nn


class NodeEmbedding(nn.Module):
    """
    NODE FEATURE EMBEDDING

    Input: Batch x Length x Dim or Batch x Length
    Output: Batch x Length x Dim

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
    """

    def __init__(self,
                 n_in: int,
                 n_out: int):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.n_in = n_in

    def forward(self,
                input_node: Optional[torch.Tensor] = None) -> Union[torch.Tensor,
                                                                    None]:
        """
        Forward node feature embedding.

        Args:
            input_node (torch.Tensor): Node features (RGB or image patches).

        Returns:
            torch.Tensor: Embedded features.
        """
        if input_node is None:
            return None

        return self.linear(input_node)


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

    def __init__(self,
                 n_out: int,
                 sigma: float):
        super().__init__()
        self.linear = nn.Linear(1, n_out, bias=False)
        self.n_out = n_out
        self.sigma = sigma

    def forward(self,
                input_coord: torch.Tensor) -> torch.Tensor:
        """
        Forward node feature embedding.

        Args:
            input_coord (torch.Tensor): Edge features ([N, 2] or [N, 3]
                coordinates array).

        Returns:
            torch.Tensor: Embedded features.
        """
        dist = torch.cdist(input_coord, input_coord)
        dist = torch.exp(-dist ** 2 / (self.sigma ** 2 * 2))
        isnan = torch.isnan(dist)
        dist = torch.where(isnan, torch.zeros_like(dist), dist)

        return self.linear(dist.unsqueeze(3))
