from typing import Optional

import torch
import torch.nn as nn


class NodeEmbedding(nn.Module):
    """
    NODE FEATURE EMBEDDING

    Input: Batch x Length x Dim
    if Batch x Length -> Batch x Length x Dim(1)
    """
    def __init__(self,
                 n_in: int,
                 n_out: int):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out, bias=False)
        self.n_in = n_in
        self.n_out = n_out

    def forward(self,
                input_node: torch.Tensor):
        if input is None:
            return None

        if self.n_in == 1:
            return self.linear(input_node.unsqueeze(2))
        else:
            return self.linear(input_node)


class EdgeEmbedding(nn.Module):
    """
    COORDINATE EMBEDDING INTO GRAPH

    Set of coordinates is used to build distance matrix which is then
    normalized using negative parabolic function.

    Args:
        n_out: Number of features to output.
    """

    def __init__(self,
                 n_out: int,
                 sigma: Optional[tuple] = int):
        super().__init__()
        self.linear = nn.Linear(1, n_out, bias=False)
        self.n_out = n_out
        self.sigma = sigma

    def forward(self,
                input_coord: torch.Tensor):
        if input_coord is None:
            return 0

        dist = torch.cdist(input_coord, input_coord)
        dist = torch.exp(-dist ** 2 / (self.sigma ** 2 * 2))
        isnan = torch.isnan(dist)
        dist = torch.where(isnan, torch.zeros_like(dist), dist)
        dist = dist.unsqueeze(3)

        return self.linear(dist)


@torch.jit.script
def gelu(x: torch.Tensor):
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        x: torch input for activation.
    """
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421356237))
