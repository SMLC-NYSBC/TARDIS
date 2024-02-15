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

from tardis_em.dist_pytorch.sparse_model.embedding import SparseEdgeEmbeddingV4
from tardis_em.dist_pytorch.sparse_model.layers import SparseDistStack


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
        knn=8,
        coord_embed_sigma=1.0,
        predict=False,
        device="cpu",
    ):
        """
        Initializes the SparseDIST.

        Args:
            n_out (int): The number of output features.
            edge_dim (int): The dimensionality of the edge features.
            num_layers (int): The number of transformer layers.
            coord_embed_sigma (int, list): The standard deviation for the edge embedding.
            predict (bool): A boolean value to decide whether to apply
                a sigmoid activation to the final output.
        """
        super(SparseDIST, self).__init__()

        self.n_out = n_out
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.knn = knn
        self.edge_sigma = coord_embed_sigma
        self.predict = predict

        self.coord_embed = SparseEdgeEmbeddingV4(
            n_out=self.edge_dim,
            sigma=self.edge_sigma,
            knn=self.knn,
            _device=device,
        )

        self.layers = SparseDistStack(
            pairs_dim=self.edge_dim,
            num_layers=self.num_layers,
            ff_factor=4,
            knn=self.knn,
        )

        self.decoder = nn.Linear(in_features=self.edge_dim, out_features=self.n_out)

    def embed_input(self, coords: torch.tensor) -> torch.tensor:
        """
        Embeds the input coordinates using the coord_embed layer.

        Args:
            coords (torch.sparse_coo_tensor): A sparse coordinate tensor
                containing the input coordinates.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor containing the embedded coordinates.
        """
        x, idx = self.coord_embed(input_coord=coords)

        return x, idx

    def forward(self, coord: torch.tensor) -> torch.tensor:
        """
        Forward pass for the SparseDIST.

        Args:
            coord (torch.tensor): A sparse coordinate tensor containing the input data.

        Returns:
            torch.sparse_coo_tensor: A sparse coordinate tensor
                representing the output from the model.
        """
        # Embed coord [n, 3] coordinates into spares tensor
        edge, idx = self.embed_input(coords=coord)  # List[Indices, Values, Shape]

        # Encode throughout the transformer layers
        edge = self.layers(edge_features=edge, indices=idx)

        # Decoder
        edge = self.decoder(edge)

        if self.predict:
            edge = torch.sigmoid(edge)

        return edge[1:, :], idx
