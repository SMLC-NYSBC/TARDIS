#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from typing import Union

import torch
from torch import nn

from tardis_em.dist_pytorch.sparse_model.modules import (
    SparsTriangularUpdate,
)
from tardis_em.dist_pytorch.model.modules import GeluFeedForward


class SparseDistStack(nn.Module):
    """
    Module that stacks multiple layers of SparseDistLayer modules.

    This class allows you to define a deep architecture where each layer is a SparseDistLayer.
    The input to each layer is the output from the previous layer,
    and the output from the final layer is the output of the SparseDistStack.
    """

    def __init__(self, pairs_dim: int, num_layers=1, ff_factor=4, knn=8):
        """
        Initializes the SparseDistStack.

        Args:
            pairs_dim (int): The dimensionality of the pairs in each layer.
            num_layers (int): The number of layers in the stack.
            ff_factor (int): The scale factor for the feed forward network in each layer.
            knn (int): The number of nearest neighbors to consider in each layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                SparseDistLayer(pairs_dim=pairs_dim, ff_factor=ff_factor, knn=knn)
            )

    def __len__(self) -> int:
        """
        Returns the number of layers in the stack.
        """
        return len(self.layers)

    def __getitem__(self, i: int):
        """
        Returns the layer at the specified index.

        Args:
            i (int): The index of the layer to return.

        Returns:
            The layer at the specified index.
        """
        return self.layers[i]

    def forward(
        self, edge_features: torch.tensor, indices: list
    ) -> Union[torch.tensor, list]:
        """
        Forward pass for the SparseDistStack.

        Args:
            edge_features (torch.sparse_coo_tensor): A sparse coordinate
                tensor containing the input data.
            indices (list): List of all indices

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing
                the output from the final layer in the stack.
        """
        for layer in self.layers:
            edge_features = layer(h_pairs=edge_features, indices=indices)

        return edge_features


class SparseDistLayer(nn.Module):
    """
    The main sparse distance layer in a PyTorch Module.

    This class implements a layer in a deep learning model that operates on sparse
    coordinate tensors. It applies updates to the edge features of the input data,
    and has placeholder methods for future updates to the node features.
    The edge updates include a row update, a column update, and a feed-forward network.
    """

    def __init__(self, pairs_dim: int, ff_factor=4, knn=8):
        """
        Initializes the SparseDistLayer.

        Args:
            pairs_dim (int): The dimensionality of the pairs in the layer.
            ff_factor (int): The scale factor for the feed forward network.
            knn (int): The number of nearest neighbors to consider in each update.
        """
        super().__init__()
        self.pairs_dim = pairs_dim
        self.channel_dim = int(pairs_dim / 4)
        if self.channel_dim <= 0:
            self.channel_dim = 1

        # ToDo Node optional features update

        # ToDO Edge optional MHA update

        # Edge triangular update
        self.row_update = SparsTriangularUpdate(
            input_dim=pairs_dim, channel_dim=self.channel_dim, axis=1, knn=knn
        )
        self.col_update = SparsTriangularUpdate(
            input_dim=pairs_dim, channel_dim=self.channel_dim, axis=0, knn=knn
        )

        # Edge GeLu FFN normalization layer
        self.pair_ffn = GeluFeedForward(
            input_dim=pairs_dim, ff_dim=pairs_dim * ff_factor
        )

    def update_nodes(self) -> torch.tensor:
        """
        Placeholder method for future node update functionality.
        """
        pass

    def update_edges(
        self, h_pairs: torch.tensor, indices: list
    ) -> Union[torch.tensor, list]:
        """
        Updates edge features by applying row and column updates,
        then applying a feed-forward network.

        Args:
            h_pairs (torch.sparse_coo_tensor): A sparse coordinate tensor
                containing the edge features to be updated.
            indices (list): List of indices

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor
                containing the updated edge features.
        """
        # ToDo Convert node features to edge shape

        # Update edge features
        h_pairs = (
            h_pairs
            + self.row_update(x=h_pairs, indices=indices)
            + self.col_update(x=h_pairs, indices=indices)
        )

        return h_pairs + self.pair_ffn(x=h_pairs)

    def forward(
        self, h_pairs: torch.tensor, indices: list
    ) -> Union[torch.tensor, list]:
        """
        Forward pass for the SparseDistLayer.

        Args:
            h_pairs (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input data.
            indices (list): List of indices

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor
                representing the output from the edge updates.
        """
        # ToDo Update node features and convert to edge shape

        # Update edge features
        h_pairs = self.update_edges(h_pairs=h_pairs, indices=indices)

        return h_pairs
