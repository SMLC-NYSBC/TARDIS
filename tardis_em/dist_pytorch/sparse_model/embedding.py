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
    This class implements a Sparse Edge Embedding mechanism, which is a module
    for computing adjacency matrices from input spatial coordinates in sparse
    tensor representation. Uses Gaussian kernels to measure the similarity
    between coordinates over multiple ranges of distances (sigma). Enables
    encoding spatial information into a sparse format suitable for downstream
    applications in neural networks.

    The module computes pairwise distances between input points, finds the
    nearest neighbors, and applies Gaussian kernels on the neighbor distances
    to construct sparse adjacency matrices augmented with multiple sigma ranges.
    """

    def __init__(self, n_knn: int, n_out: int, sigma: list):
        """
        Represents a class initialization for setting up parameters required for operations,
        including the number of nearest neighbors, output size, and a range defined by sigma.

        :param n_knn: Number of nearest neighbors used during initialization. Must be an integer.
        :type n_knn: int
        :param n_out: Number of output elements to generate. Must be an integer.
        :type n_out: int
        :param sigma: List defining the range boundaries, where the first element is the start
            and the second is the end of the range.
        :type sigma: list
        """
        super().__init__()
        self.k = n_knn
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.n_out = n_out
        self.sigma = sigma

    def forward(self, input_coord: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Performs a forward pass to compute an adjacency matrix using a Gaussian kernel
        applied to pairwise distances between input coordinates. The method calculates
        the top-k nearest neighbors for each coordinate, applies a Gaussian kernel
        function within pre-defined ranges, handles NaN values in the results, and
        constructs an adjacency matrix in sparse tensor format.

        :param input_coord: Input sparse coordinates of shape (N, ...), where N is the
            number of nodes.
        :return: A list containing:
            - `indices`: A 2D tensor defining the indices of the non-zero elements
              in the sparse tensor.
            - `values`: A 2D tensor defining the values corresponding to the indices
              in the sparse tensor.
            - `shape`: A tuple representing the shape of the sparse adjacency tensor.
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
    Represents a neural network module for performing sparse edge embedding.
    This class utilizes k-Nearest Neighbors (kNN) calculations, Gaussian kernel
    functions, and distance matrices for embedding computations. The primary
    purpose of the module is to generate embeddings for input coordinate tensors
    by applying a range of Gaussian filters over computed distances.
    """

    def __init__(self, n_out: int, sigma: list, knn: int, _device="cpu"):
        """
        Represents an initialization configuration for a neural network operation,
        defining parameters such as output neuron count, sigma range, k-Nearest Neighbors
        settings, and computational device.

        Attributes
        ----------
        n_out : int
            The number of output neurons for the operation.
        sigma : list
            A two-element list representing the range values used for torch.linspace
            to calculate a linearly spaced range of values.
        knn : int
            k-Nearest Neighbors value, indicating the k value for neighbor calculations.
        df_knn : Any or None
            Placeholder for future k-NN data processing or computation.
        _device : str
            The computation device to be utilized, such as "cpu" or "cuda".

        :param n_out: Number of output neurons.
        :type n_out: int
        :param sigma: A pair of values specifying the sigma range for calculations.
        :type sigma: list
        :param knn: The k value for k-Nearest Neighbors calculations.
        :type knn: int
        :param _device: (Optional) Device to use for computation. Defaults to "cpu".
        :type _device: str
        """
        super().__init__()
        self._range = torch.linspace(sigma[0], sigma[1], n_out)

        self.knn = knn
        self.df_knn = None
        self.n_out = n_out
        self.sigma = sigma
        self._device = _device

    def forward(self, input_coord: torch.tensor) -> Union[torch.tensor, list]:
        """
        Forward method for applying k-nearest neighbors (kNN) operations, Gaussian kernel
        calculations, and creation of indices for matrix operations. This method leverages
        KDTree for efficient neighbor searches and further processes the results to compute
        distance-based measures transformed via Gaussian kernels.

        :param input_coord: A PyTorch tensor containing input coordinates to compute kNN.
                            The tensor should be a 2D array-like structure where rows represent
                            points in space, and columns represent their respective dimensions.
        :return: A tuple containing:
                 - A PyTorch tensor of Gaussian kernel-transformed kNN distances for the given
                   points across the specified range.
                 - A list with four components:
                   1. A NumPy array (int32) indicating row indices for k-nearest distances.
                   2. A NumPy array (int32) indicating column indices for k-nearest distances.
                   3. A tuple of two integers representing the shape of the distance matrix.
                   4. A NumPy array (int32) of all neighbor pair IDs (i, j) for the kNN
                      computations.
        """
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
