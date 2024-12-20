#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from tardis_em.dist_pytorch.model.embedding import EdgeEmbedding, NodeEmbedding
from tardis_em.dist_pytorch.model.layers import DistStack
from tardis_em.utils.errors import TardisError


class BasicDIST(nn.Module):
    """
    This class implements the BasicDIST model, a graph-based transformer designed for the
    prediction of graph edges from input node and edge features. The model can handle node
    and edge embedding, layer stacking, and decoding mechanisms to provide predictions.

    The main purpose of this class is to act as a flexible and modular framework for
    processing graph-like structures with transformer-based operations. It supports
    different configurations including embeddings, number of layers, heads, and the ability
    to predict outputs using a sigmoid activation.
    """

    def __init__(
        self,
        n_out=1,
        node_input=0,
        node_dim=None,
        edge_dim=128,
        num_layers=6,
        num_heads=8,
        num_cls=None,
        rgb_embed_sigma=1.0,
        coord_embed_sigma=1.0,
        dropout_rate=0,
        structure="full",
        predict=False,
        edge_angles=False,
    ):
        """
        A class to construct a modular BasicDIST architecture incorporating node and edge
        embeddings, transformer-based layers for relational reasoning, and a final
        decoder for prediction.

        This class supports embeddings for node features and positional encodings,
        multi-head attention-based stacked layers for processing relational data,
        and an optional sigmoid activation layer for prediction.

        :param n_out: The dimensionality of the output layer of the network.
        :type n_out: int, optional
        :param node_input: The input dimension of node embeddings. If zero,
                           node embeddings are not initialized.
        :type node_input: int, optional
        :param node_dim: The output dimension of the node embeddings. If None,
                         node embeddings are not used.
        :type node_dim: int, optional
        :param edge_dim: The output dimension of the edge embeddings.
        :type edge_dim: int, optional
        :param num_layers: The number of transformer layers in the stack.
        :type num_layers: int, optional
        :param num_heads: The number of attention heads in each transformer layer.
        :type num_heads: int, optional
        :param num_cls: The number of classes to classify. Can be None for regression models.
        :type num_cls: int, optional
        :param rgb_embed_sigma: The scaling factor for RGB-based node embedding.
        :type rgb_embed_sigma: float, optional
        :param coord_embed_sigma: The scaling factor for coordinate-based edge embedding.
        :type coord_embed_sigma: float, optional
        :param dropout_rate: Dropout probability applied within transformer layers.
        :type dropout_rate: float, optional
        :param structure: The topological structure of the transformer, e.g., 'full' or other schemes.
        :type structure: str, optional
        :param predict: Boolean flag indicating whether the output predictions require
                        the sigmoid activation function.
        :type predict: bool, optional
        :param edge_angles: Boolean flag indicating whether edge embeddings should
                            account for angular relations between nodes.
        :type edge_angles: bool, optional

        :raises ValueError: If `node_dim` is not None but `node_input` is zero.
        """
        super(BasicDIST, self).__init__()

        self.n_out = n_out
        self.node_input = node_input
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_cls = num_cls
        self.node_sigma = rgb_embed_sigma
        self.edge_sigma = coord_embed_sigma
        self.dropout_rate = dropout_rate
        self.structure = structure
        self.predict = predict

        if self.node_dim is not None:
            if self.node_input > 0:
                self.node_embed = NodeEmbedding(
                    n_in=self.node_input, n_out=self.node_dim, sigma=self.node_sigma
                )

        self.coord_embed = EdgeEmbedding(n_out=self.edge_dim, sigma=self.edge_sigma)

        self.layers = DistStack(
            node_dim=self.node_dim,
            pairs_dim=self.edge_dim,
            dropout=self.dropout_rate,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            structure=self.structure,
        )

        self.decoder = nn.Linear(in_features=self.edge_dim, out_features=self.n_out)

        if self.predict:
            self.logits_sigmoid = nn.Sigmoid()

    def embed_input(
        self, coords: torch.Tensor, node_features: Optional[torch.Tensor] = None
    ):
        """
        Embeds input coordinates and optional node features using separate embedding mechanisms.

        :param coords: A tensor representing the input coordinates of shape
            [Batch x Length x Coordinate_Dim].
        :param node_features: An optional tensor representing the input node features
            of shape [Batch x Length x Feature_Dim]. If None, no node features are
            used in embedding.
        :return: A tuple containing the embedded node features and the embedded
            coordinates. The first element represents the embedded node features
            of shape [Batch x Length x Embedded_Dim], or None if `node_features`
            is not provided. The second element represents the embedded coordinates
            of shape [Batch x Length x Length x Channels].
        """
        if node_features is not None:
            """Batch x Length x Embedded_Dim"""
            x = self.node_embed(input_node=node_features)
        else:
            x = None

        """ Batch x Length x Length x Channels """
        z = self.coord_embed(input_coord=coords)
        return x, z

    def forward(
        self, coords: torch.Tensor, node_features=None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Processes input node and edge features through a transformer-based architecture
        to predict the graph edges.

        The method takes as input the coordinates and optional node features of a graph
        and embeds them. It applies transformer layers for encoding, followed by decoding
        to predict the graph edges. The predictions are based on the transformed edge
        features, optionally applying a sigmoid function for binary prediction.

        :param coords: Coordinates of the nodes in the graph.
        :param node_features: Optional features of the nodes in the graph.
        :type node_features: torch.Tensor or None
        :return: Predicted logits for the graph edges. If `predict` is True, returns
            logits processed through a sigmoid function for binary prediction.
        :rtype: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """
        node, edge = self.embed_input(coords=coords, node_features=node_features)

        if node is not None:
            """Length x Batch x Embedded_Dim"""
            node = node.transpose(0, 1)

        """ Encode throughout the transformer layers """
        _, edge = self.layers(
            node_features=node, edge_features=edge, src_key_padding_mask=None
        )

        """ Predict the graph edges """
        logits = self.decoder(edge + edge.transpose(1, 2))  # symmetries z
        logits = logits.permute(0, 3, 1, 2)

        # if self.num_cls is not None:
        #     # Batch x Length x Channels
        #     diag = np.arange(logits.shape[2])
        #     logits_cls = self.decoder_cls(edge)[:, diag, diag, :]

        if self.predict:
            # if self.num_cls is not None:
            #     # Batch x Length x Channel
            #     logits_cls = self.logits_cls_softmax(logits_cls)
            #     # Batch x Length
            #     logits_cls = torch.argmax(logits_cls, 2)
            # else:
            # Batch x Channels x Length x Length
            logits = self.logits_sigmoid(logits)

        # if self.num_cls is not None:
        #     return logits, logits_cls
        # else:
        return logits


class DIST(BasicDIST):
    """
    DIST class, a specialized subclass of BasicDIST.

    The DIST class inherits from the BasicDIST class and provides additional
    functionality or modifications as per its design. It is a flexible
    implementation that accepts various keyword arguments during instantiation
    to customize its behavior or configuration. This class is intended as part
    of a larger system and builds upon the foundational functionality provided
    by its superclass, BasicDIST. The implementation utilizes keyword arguments
    to allow dynamic initialization of class attributes or properties in a
    flexible manner.
    """

    def __init__(self, **kwargs):
        super(DIST, self).__init__(**kwargs)


class CDIST(BasicDIST):
    """
    CDIST class inherits from BasicDIST and is used for constructing a classification
    neural network. It dynamically creates layers based on input arguments and handles
    functionalities related to logits and predictions. This class serves as a foundational
    component for classification tasks, ensuring that proper configurations are enforced.
    """

    def __init__(self, **kwargs):
        super(CDIST, self).__init__(**kwargs)
        if self.num_cls is None:
            TardisError(
                "build_cdist_network",
                "tardis_em/dist",
                "Undefined num_cls parameter!",
            )

        self.decoder_cls = nn.Linear(
            in_features=self.edge_dim, out_features=self.num_cls
        )

        if self.predict:
            self.logits_sigmoid = nn.Sigmoid()
        self.logits_cls_softmax = nn.Softmax(dim=2)


def build_dist_network(network_type: str, structure: dict, prediction: bool):
    """
    Builds a DISTRIBUTED instance or semantic neural network based on the
    specified network type, structure parameters, and prediction mode.

    This function creates a network object, either `DIST` for instance
    segmentation tasks or `CDIST` for semantic segmentation tasks, depending
    on the provided `network_type`. The network is configured dynamically
    using the provided `structure` dictionary and the `prediction` flag,
    which determines whether the network will operate in prediction mode
    or not.

    :param network_type: Specifies the type of the network to be built. Accepted
        values are "instance" or "semantic". An error is raised if an unsupported
        value is provided.
    :type network_type: str
    :param structure: Dictionary containing all the configuration parameters
        required for creating the network. These parameters are passed as
        arguments when instantiating the `DIST` or `CDIST` objects.
    :type structure: dict
    :param prediction: Boolean flag indicating whether the network should be set up
        in prediction mode.
    :type prediction: bool
    :return: An instantiated network object of type `DIST` or `CDIST`. If an invalid
        `network_type` is provided, `None` is returned.
    :rtype: DIST or CDIST or None
    """
    if network_type not in ["instance", "semantic"]:
        TardisError(
            "build_dist_network",
            "tardis_em/dist",
            f"Wrong DIST network name {network_type}",
        )

    if network_type == "instance":
        return DIST(
            n_out=structure["n_out"],
            node_input=structure["node_input"],
            node_dim=structure["node_dim"],
            edge_dim=structure["edge_dim"],
            num_layers=structure["num_layers"],
            num_heads=structure["num_heads"],
            rgb_embed_sigma=structure["rgb_embed_sigma"],
            coord_embed_sigma=structure["coord_embed_sigma"],
            dropout_rate=structure["dropout_rate"],
            structure=structure["structure"],
            predict=prediction,
        )
    elif network_type == "semantic":
        return CDIST(
            n_out=structure["n_out"],
            node_input=structure["node_input"],
            node_dim=structure["node_dim"],
            edge_dim=structure["edge_dim"],
            num_layers=structure["num_layers"],
            num_heads=structure["num_heads"],
            num_cls=structure["num_cls"],
            coord_embed_sigma=structure["coord_embed_sigma"],
            dropout_rate=structure["dropout_rate"],
            structure=structure["structure"],
            predict=prediction,
        )
    else:
        return None
