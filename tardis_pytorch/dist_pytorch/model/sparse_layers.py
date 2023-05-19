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
from torch import nn

from tardis_pytorch.dist_pytorch.model.sparse_modules import SparsTriangularUpdate, SparseGeluFeedForward


class SparseDistStack(nn.Module):
    """
    Doc TBD
    """

    def __init__(
        self,
        pairs_dim: int,
        num_layers=1,
        ff_factor=4,
        knn=12
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = SparseDistLayer(
                pairs_dim=pairs_dim,
                ff_factor=ff_factor,
                k=knn
            )
            self.layers.append(layer)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int):
        return self.layers[i]

    def forward(self, edge_features: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Doc TBD
        """
        for layer in self.layers:
            edge_features = layer(h_pairs=edge_features)

        return edge_features


class SparseDistLayer(nn.Module):
    """
    MAIN SPARSE DIST LAYER
    """

    def __init__(self, pairs_dim: int, ff_factor=4, k=12):
        super().__init__()
        self.pairs_dim = pairs_dim
        self.channel_dim = int(pairs_dim / 4)
        if self.channel_dim <= 0:
            self.channel_dim = 1
        self.k = k

        # ToDo Node optional features update

        # ToDO Edge optional MHA update

        # Edge triangular update
        self.row_update = SparsTriangularUpdate(
            input_dim=pairs_dim,
            channel_dim=self.channel_dim,
            axis=1,
            k=self.k,
        )
        self.col_update = SparsTriangularUpdate(
            input_dim=pairs_dim,
            channel_dim=self.channel_dim,
            axis=0,
            k=self.k,
        )

        # Edge GeLu FFN normalization layer
        self.pair_ffn = SparseGeluFeedForward(
            input_dim=pairs_dim, ff_dim=pairs_dim * ff_factor
        )

    def update_nodes(self) -> torch.sparse_coo_tensor:
        """
        ToDo TBD
        """
        pass

    def update_edges(self, h_pairs: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Doc TBD
        """
        # ToDo Convert node features to edge shape

        # Update edge features
        h_pairs = h_pairs + self.row_update(z=h_pairs) + self.col_update(z=h_pairs)
        return h_pairs + self.pair_ffn(x=h_pairs)

    def forward(self, h_pairs: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Doc TBD
        """
        # ToDo Update node features and convert to edge shape

        # Update edge features
        h_pairs = self.update_edges(h_pairs=h_pairs)
        return h_pairs
