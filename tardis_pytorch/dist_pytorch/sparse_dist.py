#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2023                                            #
#######################################################################

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from tardis_pytorch.dist_pytorch.model.sparse_embedding import SparseEdgeEmbedding
from tardis_pytorch.dist_pytorch.model.sparse_layers import SparseDistStack
from tardis_pytorch.dist_pytorch.model.sparse_modules import SparseLinear, sparse_sigmoid
from tardis_pytorch.utils.errors import TardisError


class SparseDIST(nn.Module):
    def __init__(
        self,
        n_out=1,
        edge_dim=128,
        num_layers=6,
        coord_embed_sigma=1.0,
        predict=False,
    ):
        super(SparseDIST, self).__init__()

        self.n_out = n_out
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.edge_sigma = coord_embed_sigma
        self.predict = predict

        self.coord_embed = SparseEdgeEmbedding(n_out=self.edge_dim, sigma=self.edge_sigma)

        self.layers = SparseDistStack(
            pairs_dim=self.edge_dim,
            num_layers=self.num_layers,
        )

        self.decoder = SparseLinear(in_features=self.edge_dim, out_features=self.n_out)

    def embed_input(self, coords: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        z = self.coord_embed(input_coord=coords)
        return z

    def forward(self, coords: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        edge = self.embed_input(coords=coords)

        """ Encode throughout the transformer layers """
        edge = self.layers(edge_features=edge)

        """ Predict the graph edges """
        logits = self.decoder(edge + edge.transpose(1, 2))  # symmetries z

        if self.predict:
            logits = sparse_sigmoid(logits)

        return logits
