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

import numpy as np
from typing import Union

from scipy.spatial import KDTree
import numpy as np


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


class SparseEdgeEmbeddingV2(nn.Module):
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

    def forward(self, input_coord: torch.tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass for SparseEdgeEmbedding.

        Args:
            input_coord (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input coordinates.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing the adjacency matrix.
        """
        input_coord = input_coord.detach().requires_grad_(False)

        with torch.no_grad():
            """Calculate pairwise distances between input coordinates"""
            g_len = int(input_coord.shape[0])
            dist_ = torch.cdist(input_coord, input_coord).detach()

            """Drop too far point and keep only by top-k by row and coll"""
            k_dist_row, k_idx_row = dist_.topk(
                self.k + 1, dim=1, largest=False
            )  # Rows N x knn aka. j's for each i
            k_dist_row, k_idx_row = k_dist_row[:, 1:], k_idx_row[:, 1:]
            # k_dist_col, k_idx_col = dist_.topk(self.k+1 , dim=0, largest=False)  # Colums knn x N aka i's for each j
            # k_dist_col, k_idx_col = k_dist_col[1:, :], k_idx_col[1:, :]

            k_dist_row = k_dist_row.flatten()

            k_idx_row_flat = torch.zeros((g_len * self.k, 2), dtype=torch.int16)
            # k_idx_col_flat = torch.zeros((g_len*self.k, 2), dtype=torch.int16)
            df_idx = torch.Tensor(
                np.repeat(np.arange(0, g_len), self.k).astype(np.int16)
            )

            k_idx_row_flat[:, 1] = k_idx_row.flatten()
            k_idx_row_flat[:, 0] = df_idx
            # k_idx_col_flat[:, 0] = k_idx_col.transpose(1,0).flatten()
            # k_idx_col_flat[:, 1] = df_idx

            # k_idx = torch.concatenate((k_idx_row_flat, k_idx_col_flat))

            """[1] - Apply Gaussian kernel function to the top-k distances"""
            k_dist_range = torch.zeros(
                (k_dist_row.shape[0], len(self._range)), device=dist_.device
            )

            for id_, i in enumerate(self._range):
                k_dist_range[:, id_] = torch.exp(
                    -(k_dist_row**2) / (i**2 * 2)
                )  # M x knn

            # Replace any NaN values with zero
            isnan = torch.isnan(k_dist_range)
            k_dist_range = torch.where(
                isnan, torch.zeros_like(k_dist_range), k_dist_range
            )

        # List
        # [0] ij_idx [B, M, 2] (int) row_idx
        # [1] All ij [B, M, O] (float_64)
        # [2] Shape (B, Row, Col, Ch)
        return k_dist_range.unsqueeze(0), [
            k_idx_row_flat,
            (1, g_len, g_len, self.n_out),
        ]


class SparseEdgeEmbeddingV3(nn.Module):
    """
    Module for Sparse Edge Embedding.

    This class is responsible for computing a sparse adjacency matrix
    with edge weights computed using a Gaussian kernel function over
    the distances between input coordinates.
    """

    def __init__(self, n_out: int, sigma: list):
        """
        Initializes the SparseEdgeEmbedding.

        Args:
            n_out (int): The number of output channels.
            sigma (list): The range of sigma values for the Gaussian kernel.
        """
        super().__init__()
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.n_out = n_out
        self.sigma = sigma

    def forward(self, input_coord: torch.tensor) -> Union[torch.tensor, list]:
        """
        Embed input coordinates.

        The embedding is processed by firstly extracting every ij elements from
        distance matrix, and performing gaussian kernel function.
        As a second step indices for all ij elements are extracted, followed by
        ij id's for row and columns.

        - Long tensor [Batch, M, Channel] <- M is every ij element
        - Indices:
            - All M_ij [[M, 1], [M, 1]] <- Correspond to i and j sorted along rows
            - Rows: [[[0, 1, 2], ...], [[234, 12, 1], ...]
                Two list corresponding to M_ij. First indicates column M_ij id's per row,
                the second one, correspond to M_ij id's for each colum per row.
            - Column: [[[0, 1, 2], ...], [[234, 12, 1], ...]
                Two list corresponding to M_ij. First indicates row M_ij id's per column,
                the second one, correspond to M_ij id's for each row per column.
            - Shape: Final shape of 2D graph

        Args:
         input_coord (torch.tensor): 2D or 3D array of coordinates

        Return:
            torch.tensor, idx: Long tensor with all ij elements, and list of all indices.
        """
        if isinstance(input_coord, np.ndarray):
            input_coord = torch.Tensor(input_coord)

        # Run on CPU
        device = input_coord.device
        input_coord = input_coord.cpu().detach()

        with torch.no_grad():
            """Calculate pairwise distances between input coordinates"""
            g_len = int(input_coord.shape[0])
            _dist = torch.cdist(input_coord, input_coord).detach()
            _shape = (g_len, g_len, self.n_out)  # [4] Original 2D shape to reconstruct

            """[0] - Get all IDX to reconstruct"""
            mask = torch.exp(-(_dist**2) / (self._range[-1] ** 2 * 2)) < 0.9

            _dist[torch.where(mask)] = 0
            _dist.fill_diagonal_(0)

            indices = torch.where(
                _dist > 0
            )  # every ij elemtent from distance embedding

            """[2-3] Get Col/Row wise indices for ij element"""
            # Columns list
            unique_elements, inverse_indices = torch.unique(
                indices[0], return_inverse=True
            )
            # Number of elements per row
            counts = torch.bincount(inverse_indices)

            # get row-wise indices
            starts = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=_dist.device),
                    counts.cumsum(0)[:-1],
                )
            )
            indices_ = [
                torch.arange(start, start + count)
                for start, count in zip(starts, counts)
            ]
            row_indices = [
                idx.tolist() for element, idx in zip(unique_elements, indices_)
            ]

            indices_ = [
                indices[1][start : start + count]
                for start, count in zip(starts, counts)
            ]
            row_idx = [idx.tolist() for element, idx in zip(unique_elements, indices_)]
            row_indices = [row_indices, row_idx]

            # get col-wise indices
            col_indices = [[] for x in range(len(row_idx))]
            col_idx = [[] for x in range(len(row_idx))]

            for _id, (row, col) in enumerate(zip(indices[0], indices[1])):
                col_indices[col.item()].append(_id)  # Colum id from M
                col_idx[col.item()].append(row.item())  # Column triangle ID

            col_indices = [col_indices, col_idx]

            """[1] - Apply Gaussian kernel function to the top-k distances"""
            indices = list(
                (
                    list(indices[0].cpu().detach().numpy()),
                    list(indices[1].cpu().detach().numpy()),
                )
            )
            _dist = _dist[indices]
            k_dist_range = torch.zeros(
                (len(_dist), len(self._range)), device=_dist.device
            )

            # Apply Gaussian kernel function to all ij elements
            for id_, i in enumerate(self._range):
                k_dist_range[:, id_] = torch.exp(-(_dist**2) / (i**2 * 2))

        return k_dist_range.unsqueeze(0).to(device), [
            indices,
            row_indices,
            col_indices,
            _shape,
        ]


class SparseEdgeEmbeddingV4(nn.Module):
    """
    Module for Sparse Edge Embedding.

    This class is responsible for computing a sparse adjacency matrix
    with edge weights computed using a Gaussian kernel function over
    the distances between input coordinates.
    """

    def __init__(self, n_out: int, sigma: list, knn: int, _device):
        """
        Initializes the SparseEdgeEmbedding.

        Args:
            n_out (int): The number of output channels.
            sigma (list): The range of sigma values for the Gaussian kernel.
        """
        super().__init__()
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.knn = knn
        self.n_out = n_out
        self.sigma = sigma
        self._device = _device

    def forward(self, input_coord: torch.tensor) -> Union[torch.tensor, list]:
        with torch.no_grad():
            # Get all ij element from row and col
            input_coord = input_coord.cpu().detach().numpy()
            tree = KDTree(input_coord)
            distances, indices = tree.query(input_coord, self.knn, p=1)

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
                    np.pad(c, (0, self.knn - len(c)))
                    if len(c) <= self.knn
                    else c[np.argsort(M[c - 1])[: self.knn]]
                    for c in [
                        np.where(all_ij_id[:, 1] == i)[0] + 1
                        for i in range(len(input_coord))
                    ]
                ]
            )

            col_idx = np.repeat(col_idx, self.knn, axis=0)
            col_idx = np.vstack((np.repeat(0, self.knn), col_idx))

            M = torch.from_numpy(np.pad(M, (1, 0)))
            # Prepare tensor for storing range of distances
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

        # # symetry indexes
        # # Convert to structured arrays
        # b = np.stack((all_ij_id[:,1], all_ij_id[:, 0])).T

        # a_structured = np.ascontiguousarray(all_ij_id).view(np.dtype((np.void, all_ij_id.dtype.itemsize * all_ij_id.shape[1])))
        # b_structured = np.ascontiguousarray(b).view(np.dtype((np.void, b.dtype.itemsize * b.shape[1])))

        # # Get the indices of duplicated rows
        # sum_a_value = np.where(np.in1d(a_structured, b_structured))[0]
        # sum_b_value = np.where(np.in1d(b_structured, a_structured))[0]

        # add_value = np.setdiff1d(b_structured, a_structured).view(all_ij_id.dtype).reshape(-1, all_ij_id.shape[1])

        return k_dist_range.to(self._device), [
            row_idx.astype(np.int32),
            col_idx.astype(np.int32),
            (n, n),
            all_ij_id.astype(np.int32),
        ]
