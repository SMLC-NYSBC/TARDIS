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

from tardis_pytorch.dist_pytorch.sparse_model.embedding import SparseEdgeEmbedding
from tardis_pytorch.dist_pytorch.sparse_model.layers import SparseDistStack
from tardis_pytorch.dist_pytorch.sparse_model.modules import (
    SparseLinear,
    sparse_sigmoid,
    sparse_operation,
)


class SparseDIST(nn.Module):
    """
    Sparse Distance Transformer model in a PyTorch Module.

    This class implements a transformer model that can handle sparse data efficiently.
    It first uses a SparseEdgeEmbedding layer to embed the input coordinate tensor into
    a higher-dimensional space. The SparseDistStack then applies several transformer layers
    to this embedded tensor.
    Lastly, a SparseLinear layer decodes the result into the final output.
    The forward pass of the model can optionally apply a sigmoid function
    to the final output if the predict attribute is set to True.
    """

    def __init__(
        self,
        n_out=1,
        edge_dim=128,
        num_layers=6,
        knn=12,
        coord_embed_sigma=1.0,
        predict=False,
    ):
        """
        Initializes the SparseDIST.

        Args:
            n_out (int): The number of output features.
            edge_dim (int): The dimensionality of the edge features.
            num_layers (int): The number of transformer layers.
            knn (int): The number of nearest neighbors to consider in each update.
            coord_embed_sigma (int, list): The standard deviation for the edge embedding.
            predict (bool): A boolean value to decide whether to apply a sigmoid activation to the final output.
        """

        super(SparseDIST, self).__init__()

        self.n_out = n_out
        self.edge_dim = edge_dim
        self.knn = knn
        self.num_layers = num_layers
        self.edge_sigma = coord_embed_sigma
        self.predict = predict

        self.coord_embed = SparseEdgeEmbedding(
            n_out=self.edge_dim, sigma=self.edge_sigma, n_knn=self.knn
        )

        self.layers = SparseDistStack(
            pairs_dim=self.edge_dim,
            num_layers=self.num_layers,
            ff_factor=4,
            knn=self.knn,
        )

        self.decoder = SparseLinear(in_features=self.edge_dim, out_features=self.n_out)

    def embed_input(self, coords: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Embeds the input coordinates using the coord_embed layer.

        Args:
            coords (torch.sparse_coo_tensor): A sparse coordinate tensor containing the input coordinates.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor containing the embedded coordinates.
        """
        x = self.coord_embed(input_coord=coords)
        return x

    def forward(self, coords: torch.tensor) -> torch.sparse_coo_tensor:
        """
        Forward pass for the SparseDIST.

        Args:
            coords (torch.tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor representing the output from the model.
        """
        # Embed coord [n, 3] coordinates into spares tensor
        edge = self.embed_input(coords=coords)  # List[Indices, Values, Shape]

        # Encode throughout the transformer layers
        edge = self.layers(edge_features=edge)  # List[Indices, Values, Shape]

        # Predict the graph edges
        edge = self.decoder(
            sparse_operation(
                edge,
                sparse_operation(edge, knn=self.knn, op="rowcol_transpose"),
                op="sum",
            )
        )

        if self.predict:
            edge = sparse_sigmoid(edge)

        return torch.sparse_coo_tensor(edge[0], edge[1], edge[2]).to_dense()
