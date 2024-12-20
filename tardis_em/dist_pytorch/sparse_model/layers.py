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
    Represents a stack of SparseDistLayer modules. Each layer in the stack processes input data
    based on pairwise features and nearest neighbor interactions. The stack enables cascading
    computations in multiple layers to enhance the representation of the input.

    This class is designed for processing pairwise relationships with sparse data structures,
    particularly in machine learning frameworks where modular and scalable architecture is needed.
    It allows easy modification of the number of layers and their properties.
    """

    def __init__(self, pairs_dim: int, num_layers=1, ff_factor=4, knn=8):
        """
        Initializes the class with specified parameters and creates a module list containing
        multiple SparseDistLayer instances. Each layer is initialized with the specified
        parameters for pairs dimension, feed-forward factor, and k-nearest neighbors.

        :param pairs_dim: The dimension of the input pairwise data.
        :type pairs_dim: int
        :param num_layers: The number of layers to instantiate. Defaults to 1.
        :type num_layers: int, optional
        :param ff_factor: The multiplication factor for the feed-forward layer. Defaults to 4.
        :type ff_factor: int, optional
        :param knn: The number of k-nearest neighbors to use. Defaults to 8.
        :type knn: int, optional
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
        Retrieves the layer at the given index from the collection of layers.

        This method provides access to a specific layer object, based on the provided
        index. The index corresponds to the position of the layer within the internal
        list. If the index is out of the range of the collection, an exception will be
        raised.

        :param i: The index of the layer to retrieve.
        :type i: int
        :return: The layer object at the specified index.
        :rtype: Any
        """
        return self.layers[i]

    def forward(
        self, edge_features: torch.tensor, indices: list
    ) -> Union[torch.tensor, list]:
        """
        Processes the given edge features and indices through a sequence of layers, updating the edge features
        at each step by passing them through the corresponding layer along with the provided indices. The
        resulting edge features after processing through all layers are returned.

        :param edge_features: A tensor containing the features associated with the edges. Each layer uses this
            tensor during its processing step to compute the updated edge features.
        :param indices: A list of indices corresponding to the edges. These indices are used during the
            computation in each layer when processing the edge features.
        :return: A tensor containing the updated edge features after processing through all layers sequentially.
        """
        for layer in self.layers:
            edge_features = layer(h_pairs=edge_features, indices=indices)

        return edge_features


class SparseDistLayer(nn.Module):
    """
    SparseDistLayer is a neural network module designed for updating sparse edge
    features, leveraging nearest neighbors, feed-forward networks, and triangular
    updates. The main purpose of this layer is to process and transform sparse
    relationships while effectively handling computational efficiencies for
    large-scale data representations in sparse tensor formats.

    This class serves as an essential component in advanced neural network
    architectures and is particularly tailored for sparse data processing,
    supporting updates for both edges and potentially nodes in future iterations.
    """

    def __init__(self, pairs_dim: int, ff_factor=4, knn=8):
        """
        Initializes an object with configuration for edge updates and feed-forward normalization.

        :param pairs_dim: The dimensionality of input pairs.
        :param ff_factor: The factor determining the hidden layer dimensionality in the
                          feed-forward network. Defaults to 4.
        :param knn: The number of nearest neighbors for sparse triangular updates. Defaults to 8.
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
        Updates the edge features by applying transformations to the input node
        feature pair tensor. These transformations include row-wise, column-wise
        updates, and a feed-forward network applied on the node pairs. The method
        returns the modified tensor, representing updated edge features.

        :param h_pairs: Tensor representing the features of node pairs that
            need to be updated.
        :param indices: List of indices used for performing updates on the
            edge features.
        :return: Tensor representing the updated edge features, which includes
            the applied transformations.
        :rtype: Union[torch.tensor, list]
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
        Updates the features of input nodes and reshapes them to have the required edge
        shape. This method first updates the edge features based on the given pairs and
        indices, then returns the updated features.

        :param h_pairs: A tensor representing the input node features. Dimensions and
            meaning depend on the tensor's context within the model.
        :param indices: A list of indices used to update the edge features. The indices
            specify how the input tensor `h_pairs` is altered.
        :return: A tensor or list with updated edge features, containing the transformed
            input node values based on the specified indices. The return type is dependent
            on the implementation of `update_edges`.
        """
        # ToDo Update node features and convert to edge shape

        # Update edge features
        h_pairs = self.update_edges(h_pairs=h_pairs, indices=indices)

        return h_pairs
