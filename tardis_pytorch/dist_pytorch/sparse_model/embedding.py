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
    Module for Sparse Edge Embedding.

    This class is responsible for computing a sparse adjacency matrix
    with edge weights computed using a Gaussian kernel function over
    the distances between input coordinates.
    """

    def __init__(self, n_knn: int, n_out: int, sigma: list):
        """
        Initializes the SparseEdgeEmbedding.

        Args:
            n_knn (int): The number of nearest neighbors to consider for each point.
            n_out (int): The number of output channels.
            sigma (list): The range of sigma values for the Gaussian kernel.
        """
        super().__init__()
        self.k = n_knn
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.n_out = n_out
        self.sigma = sigma

    def forward(self, input_coord: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass for SparseEdgeEmbedding.

        Args:
            input_coord (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input coordinates.

        Returns:
            torch.sparse_coo_tensor: Asparse coordinate tensor representing the adjacency matrix.
        """
        # Calculate pairwise distances between input coordinates
        g_len = input_coord.shape[0]
        dist_ = torch.cdist(input_coord, input_coord)

        # Get the top-k nearest neighbors and their indices
        k_dist, k_idx = dist_.topk(self.k, dim=-1, largest=False)

        # Prepare tensor for storing range of distances
        k_dist_range = torch.zeros(
            (g_len, self.k, len(self._range)), device=dist_.device
        )

        # Apply Gaussian kernel function to the top-k distances
        for id_, i in enumerate(self._range):
            k_dist_range[:, :, id_] = torch.exp(-(k_dist**2) / (i**2 * 2))

        # Replace any NaN values with zero
        isnan = torch.isnan(k_dist_range)
        k_dist_range = torch.where(isnan, torch.zeros_like(k_dist_range), k_dist_range)

        # Prepare indices for constructing the adjacency matrix
        row = (
            torch.arange(g_len, device=dist_.device)
            .unsqueeze(-1)
            .repeat(1, self.k)
            .view(-1)
        )
        col = k_idx.view(-1)
        batch = torch.zeros_like(col)

        indices = torch.cat(
            [batch.unsqueeze(0), row.unsqueeze(0), col.unsqueeze(0)], dim=0
        )

        # Prepare values for constructing the adjacency matrix
        values = k_dist_range.view(-1, self.n_out)

        # Construct the adjacency matrix as a sparse coordinate tensor
        adj_matrix = torch.sparse_coo_tensor(
            indices, values, (1, g_len, g_len, self.n_out)
        )

        return adj_matrix
