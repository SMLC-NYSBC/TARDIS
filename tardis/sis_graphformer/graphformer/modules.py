import torch
import torch.distributions
import torch.nn as nn


class DistEmbedding(nn.Module):
    """
    COORDINATE EMBEDDING INTO GRAPH

    Set of coordinates is used to build distance matrix which is then
    normalized using negative parabolic function.

    Args:
        n_out: Number of features to output.
        sigma: Normalize factor for the parabolic function
    """

    def __init__(self,
                 n_out: int,
                 sigma=16):
        super().__init__()
        self.sigma = sigma
        self.linear = nn.Linear(1, n_out, bias=False)

    def forward(self, x):
        if x is None:
            return 0

        dist = torch.cdist(x, x)
        kernel = torch.exp(-dist ** 2 / (self.sigma ** 2 * 2))  # TODO scaling by sigma not linear
        isnan = torch.isnan(kernel)
        kernel = torch.where(isnan, torch.zeros_like(kernel), kernel)
        kernel = kernel.unsqueeze(3)

        return self.linear(kernel)


@torch.jit.script
def gelu(x: torch.Tensor):
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        x: torch input for activation.
    """
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421356237))
