from math import sqrt
from typing import Optional

import torch
import torch.nn as nn


class DistEmbedding(nn.Module):
    """
    COORDINATE EMBEDDING INTO GRAPH

    Set of coordinates is used to build distance matrix which is then
    normalized using negative parabolic function.

    Args:
        n_out: Number of features to output.
    """

    def __init__(self,
                 n_out: int,
                 dist: bool,
                 sigma: Optional[tuple] = int):
        super().__init__()
        self.linear = nn.Linear(1, n_out, bias=False)
        self.n_out = n_out
        self.sigma = sigma
        self.dist = dist

    def forward(self,
                x: torch.Tensor):
        if x is None:
            return 0

        x = torch.cdist(x, x)

        if self.dist:
            x = torch.exp(-x ** 2 / (self.sigma ** 2 * 2))

            isnan = torch.isnan(x)
            x = torch.where(isnan, torch.zeros_like(x), x)
        else:
            size = x.shape[1]
            kernel = torch.zeros((1, size, size)).to(x.device)

            for i in range(size):
                kernel[0, i, i] = 1

        x = x.unsqueeze(3)

        return self.linear(x)


@torch.jit.script
def gelu(x: torch.Tensor):
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        x: torch input for activation.
    """
    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2)))
