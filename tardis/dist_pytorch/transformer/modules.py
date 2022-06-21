import math
import torch
import torch.nn as nn
import math


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
                 sigma: int,
                 dist: bool):
        super().__init__()
        self.linear = nn.Linear(1, n_out, bias=False)
        self.n_out = n_out
        self.sigma = sigma
        self.dist = dist

    def forward(self, 
                x: torch.Tensor):
        if x is None:
            return 0

        if self.dist:
            dist = torch.cdist(x, x)
            kernel = torch.exp(-dist ** 2 / (self.sigma ** 2 * 2))
            isnan = torch.isnan(kernel)
            kernel = torch.where(isnan, torch.zeros_like(kernel), kernel)
            kernel = kernel.unsqueeze(3)

            return self.linear(kernel)
        else:
            size = x.shape[1]
            kernel = torch.zeros((1, size, size)).to(x.device) # may explode!!!!!!

            for i in range(size):
                kernel[0, i, i] = 1

            kernel = kernel.unsqueeze(3)

            return  self.linear(kernel)


@torch.jit.script
def gelu(x: torch.Tensor):
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        x: torch input for activation.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))
