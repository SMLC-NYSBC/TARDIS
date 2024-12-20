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
    SparseDIST is a neural network model designed to process sparse data using
    transformer-based layers. It embeds input coordinates, encodes them using
    multiple layers, and optionally applies a sigmoid activation for prediction
    tasks. SparseDIST is highly optimized for operations on sparse tensors,
    making it suitable for large-scale data processing tasks requiring a sparse
    data representation.
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
        This class implements a SparseDIST model with customizable parameters such as the
        number of output features, edge dimensionality, number of layers in the network,
        k-nearest neighbors (kNN) configuration, and device specification. It leverages
        a SparseEdgeEmbeddingV4 instance for edge embeddings and a SparseDistStack
        component for stacking the model's layers effectively. The model also supports
        prediction with a decoder stage implemented as a linear layer.

        :param n_out: Number of output features for the decoder.
        :type n_out: int
        :param edge_dim: Dimensionality of edge embeddings.
        :type edge_dim: int
        :param num_layers: Number of stacked layers in the SparseDistStack.
        :type num_layers: int
        :param knn: Number of nearest neighbors to consider for the kNN graph.
        :type knn: int
        :param coord_embed_sigma: Sigma value for coordinate embeddings in the SparseEdgeEmbeddingV4.
        :type coord_embed_sigma: float
        :param predict: Whether the model should operate in prediction mode.
        :type predict: bool
        :param device: The device on which the model computations will run (e.g., 'cpu', 'cuda').
        :type device: str
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
        Embeds the input coordinates using a predefined embedding method.

        The method takes input coordinates and processes them through
        a coordinate embedding function. The function returns the
        embedded representation of the input coordinates alongside
        an index or related metadata generated during the embedding
        process.

        :param coords: The input coordinates to be embedded.

        :return: A tuple where the first element is the embedded
            coordinates and the second is the associated index or
            metadata.
        """
        x, idx = self.coord_embed(input_coord=coords)

        return x, idx

    def forward(self, coord: torch.tensor) -> torch.tensor:
        """
        This function processes spatial coordinates through embedding, encoding, and decoding
        steps in order to produce a sparse tensor. The input coordinates are first embedded
        into a sparse tensor representation. Subsequently, the data is encoded using transformer
        layers and further processed by the decoder network. If the `predict` attribute is set
        to True, a sigmoid activation is applied to the output to provide predictions in the
        range of [0, 1]. Finally, the function returns the processed edge data and the indices
        from the embedding step.

        :param coord: A tensor of shape [n, 3] representing n spatial coordinates. Coordinates
                      are typically three-dimensional.
        :return: A tuple containing:
                 - A tensor representing the processed edge data, starting from the second row:
                   [1:, :].
                 - The indices from the initial embedding step.
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
