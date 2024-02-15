#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

import torch
import torch.nn as nn

import numpy as np
from typing import Union

from scipy.spatial import KDTree


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
            input_coord (torch.sparse_coo_tensor): A sparse coordinate tensor
                containing the input coordinates.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing the adjacency matrix.
        """
        with torch.no_grad():
            # Calculate pairwise distances between input coordinates
            g_len = int(input_coord.shape[0])
            dist_ = torch.cdist(input_coord, input_coord).detach()

            # Get the top-k nearest neighbors and their indices
            k_dist, k_idx = dist_.topk(self.k, dim=-1, largest=False)

            # Prepare tensor for storing range of distances
            k_dist_range = torch.zeros(
                (g_len, self.k, len(self._range)), device=dist_.device
            )

            # Apply Gaussian kernel function to the top-k distances
            for id_, i in enumerate(self._range):
                dist_range = torch.exp(-(k_dist**2) / (i**2 * 2))

                k_dist_range[:, :, id_] = torch.where(dist_range > 0.1, dist_range, 0)

            # Replace any NaN values with zero
            isnan = torch.isnan(k_dist_range)
            k_dist_range = torch.where(
                isnan, torch.zeros_like(k_dist_range), k_dist_range
            )

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

        return [
            indices.detach(),
            values.requires_grad_(),
            (1, g_len, g_len, self.n_out),
        ]


class SparseEdgeEmbeddingV4(nn.Module):
    """
    Module for Sparse Edge Embedding.

    This class is responsible for computing a sparse adjacency matrix
    with edge weights computed using a Gaussian kernel function over
    the distances between input coordinates.
    """

    def __init__(self, n_out: int, sigma: list, knn: int, _device="cpu"):
        """
        Initializes the SparseEdgeEmbedding.

        Args:
            n_out (int): The number of output channels.
            sigma (list): The range of sigma values for the Gaussian kernel.
        """
        super().__init__()
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.knn = knn
        self.df_knn = None
        self.n_out = n_out
        self.sigma = sigma
        self._device = _device

    def forward(self, input_coord: torch.tensor) -> Union[torch.tensor, list]:
        with torch.no_grad():
            # Get all ij element from row and col
            input_coord = input_coord.cpu().detach().numpy()
            tree = KDTree(input_coord)
            if len(input_coord) < self.knn:
                self.df_knn = int(self.knn)
                self.knn = len(input_coord)

            distances, indices = tree.query(input_coord, self.knn, p=2)

            n = len(input_coord)
            M = distances.flatten()

            all_ij_id = np.array(
                (np.repeat(np.arange(n), self.knn), indices.flatten())
            ).T

            # Row-Wise M[ij] index
            row_idx = np.repeat(
                np.arange(0, len(M)).reshape(len(input_coord), self.knn),
                self.knn,
                axis=0,
            ).reshape(len(M), self.knn)
            row_idx = np.vstack((np.repeat(0, self.knn), row_idx))

            # Column-Wise M[ij] index
            col_idx = np.array(
                [
                    (
                        np.pad(c, (0, self.knn - len(c)))
                        if len(c) <= self.knn
                        else c[np.argsort(M[c - 1])[: self.knn]]
                    )
                    for c in [
                        np.where(all_ij_id[:, 1] == i)[0] + 1
                        for i in range(len(input_coord))
                    ]
                ]
            )

            col_idx = np.repeat(col_idx, self.knn, axis=0)
            col_idx = np.vstack((np.repeat(0, self.knn), col_idx))

            M = torch.from_numpy(np.pad(M, (1, 0)))
            # Prepare tensor for storing a range of distances
            k_dist_range = torch.zeros((len(M), len(self._range)))

            # Apply Gaussian kernel function to the top-k distances
            for id_, i in enumerate(self._range):
                k_dist_range[:, id_] = torch.exp(-(M**2) / (i**2 * 2))
            k_dist_range[0, :] = 0

            # Replace any NaN values with zero
            isnan = torch.isnan(k_dist_range)
            k_dist_range = torch.where(
                isnan, torch.zeros_like(k_dist_range), k_dist_range
            )

        if self.df_knn is not None:
            self.knn = int(self.df_knn)
            self.df_knn = None

        return k_dist_range.to(self._device), [
            row_idx.astype(np.int32),
            col_idx.astype(np.int32),
            (n, n),
            all_ij_id.astype(np.int32),
        ]
