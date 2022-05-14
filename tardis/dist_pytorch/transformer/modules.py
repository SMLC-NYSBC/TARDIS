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
                 sigma: int):
        super().__init__()
        self.linear = nn.Linear(1, n_out, bias=False)
        self.n_out = n_out
        self.sigma = sigma

    def forward(self, x):
        if x is None:
            return 0

        dist = torch.cdist(x, x)
        kernel = torch.exp(-dist ** 2 / (self.sigma ** 2 * 2))
        isnan = torch.isnan(kernel)
        kernel = torch.where(isnan, torch.zeros_like(kernel), kernel)
        kernel = kernel.unsqueeze(3)

        return self.linear(kernel)

        # # Build kernels
        # shape = x.shape
        # kernels = torch.zeros((shape[0], shape[1], shape[1], self.n_out))
        # dist = torch.cdist(x, x)

        # # Scaling by sigma
        # sigma = self.linear(torch.Tensor([self.n_out]).to(x.device))

        # for id, s in enumerate(sigma):
        #     kernel = torch.exp(-dist ** 2 / (s ** 2 * 2))
        #     isnan = torch.isnan(kernel)
        #     kernel = torch.where(isnan, torch.zeros_like(kernel), kernel)
        #     kernels[:, :, :, id] = kernel

        # return kernels.to(x.device)


@torch.jit.script
def gelu(x: torch.Tensor):
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        x: torch input for activation.
    """
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421356237))
