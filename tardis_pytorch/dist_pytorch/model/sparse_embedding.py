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


class SparseEdgeEmbedding(nn.Module):
    """
    COORDINATE EMBEDDING INTO SPARSE GRAPH

    Set of coordinates is used to build k-NN graph which is then
    normalized using negative parabolic function.

    Input: Batch x Length x Dim
    Output: SparseTensor with shape [Length x Length]

    Args:
        k (int): Number of nearest neighbors.
        n_out (int): Number of features to output.
        sigma (int, optional tuple): Sigma value for an exponential function is
            used to normalize distances.
    """

    def __init__(self, k: int, n_out: int, sigma: list):
        super().__init__()
        self.k = k
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.n_out = n_out
        self.sigma = sigma

    def forward(self, input_coord: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Forward node feature embedding.

        Args:
            input_coord (torch.Tensor): Edge features ([B, N, 2] or [B, N, 3]
                coordinates array).

        Returns:
            torch.Tensor: Embedded features.
        """
        g_len = input_coord.shape[0]
        dist_ = torch.cdist(input_coord, input_coord)
        k_dist, k_idx = dist_.topk(self.k, dim=-1, largest=False)

        k_dist_range = torch.zeros(
            (g_len, self.k, len(self._range)), device=dist_.device
        )

        for id_, i in enumerate(self._range):
            k_dist_range[:, :, id_] = torch.exp(-(k_dist**2) / (i**2 * 2))

        isnan = torch.isnan(k_dist_range)
        k_dist_range = torch.where(isnan, torch.zeros_like(k_dist_range), k_dist_range)

        row = (
            torch.arange(g_len, device=dist_.device)
            .unsqueeze(-1)
            .repeat(1, self.k)
            .view(-1)
        )
        col = k_idx.view(-1)

        indices = torch.cat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
        values = k_dist_range.view(-1, self.n_out)
        adj_matrix = torch.sparse_coo_tensor(
            indices, values, (g_len, g_len, self.n_out)
        )

        return adj_matrix
