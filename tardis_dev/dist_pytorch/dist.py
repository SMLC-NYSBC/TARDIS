from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tardis.dist_pytorch.transformer.modules import (EdgeEmbedding,
                                                     NodeEmbedding)
from tardis_dev.dist_pytorch.model.layers import DistStack


class BasicDIST(nn.Module):
    """
    General DIST FOR DIMENSIONLESS INSTANCE SEGMENTATION TRANSFORMER

    Args:
        n_out (int): Number of channels in the output layer.
        node_input (int): Length of the flattened image file.
        node_dim (int, optional): In features of image for linear transformation.
        edge_dim (int): In feature of coord for linear transformation.
        num_layers (int): Number of DIST layers to initialize.
        num_heads (int): Number of heads for MHA.
        num_cls (int): Number of predicted classes.
        coord_embed_sigma (float): Sigma value used to embed coordinate distance
            features.
        dropout_rate (float): Dropout factor used in MHA dropout layer.
        structure (str): DIST network structure.
                Options: (full, traing, dualtriang, quad, attn)
        predict (bool): If True sigmoid output.
    """

    def __init__(self,
                 n_out=1,
                 node_input=0,
                 node_dim=None,
                 edge_dim=128,
                 num_layers=6,
                 num_heads=8,
                 num_cls=None,
                 coord_embed_sigma: Optional[tuple] = 1.0,
                 dropout_rate=0,
                 structure='full',
                 predict=False):
        super(BasicDIST, self).__init__()

        self.n_out = n_out
        self.node_input = node_input
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_cls = num_cls
        self.sigma = coord_embed_sigma
        self.dropout_rate = dropout_rate
        self.structure = structure
        self.predict = predict

        if self.node_dim is not None:
            if self.node_input > 0:
                self.node_embed = NodeEmbedding(n_in=self.node_input,
                                                n_out=self.node_dim)

        self.coord_embed = EdgeEmbedding(n_out=self.edge_dim,
                                         sigma=self.sigma)

        self.layers = DistStack(node_dim=self.node_dim,
                                pairs_dim=self.edge_dim,
                                dropout=self.dropout_rate,
                                num_layers=self.num_layers,
                                num_heads=self.num_heads,
                                structure=self.structure)

        self.decoder = nn.Linear(in_features=self.edge_dim,
                                 out_features=self.n_out)

        if self.predict:
            self.logits_sigmoid = nn.Sigmoid()

    def embed_input(self,
                    coords: torch.Tensor,
                    node_features: Optional[torch.Tensor] = None):
        """
        Embedding features

        Args:
            coords (torch.Tensor): Coordinate features.
            node_features (torch.Tensor, optional): Optional Node features.

        Returns:
            torch.tensor: Embedded features for prediction.
        """
        if node_features is not None:
            """ Batch x Length x Embedded_Dim """
            x = self.node_embed(input_node=node_features)
        else:
            x = None

        """ Batch x Length x Length x Channels """
        z = self.coord_embed(input_coord=coords)
        return x, z

    def forward(self,
                coords: torch.Tensor,
                node_features: Optional[torch.Tensor] = None):
        """
        Forward DIST model.

        Args:
            coords (torch.Tensor): Coordinates input of a shape
                [Batch x Channels x Length].
            node_features (torch.Tensor, optional): Image patch input of a shape
                [Batch x Length x Dimensions].
        """
        node, edge = self.embed_input(coords=coords,
                                      node_features=node_features)

        if node is not None:
            """ Length x Batch x Embedded_Dim """
            node = node.transpose(0, 1)

        """ Encode throughout the transformer layers """
        _, edge = self.layers(node_features=node,
                              edge_features=edge,
                              src_key_padding_mask=None)

        """ Predict the graph edges """
        logits = self.decoder(edge + edge.transpose(1, 2))  # symmetries z
        logits = logits.permute(0, 3, 1, 2)

        if self.num_cls is not None:
            # Batch x Length x Channels
            diag = np.arange(logits.shape[2])
            logits_cls = self.decoder_cls(edge)[:, diag, diag, :]

        if self.predict:
            if self.num_cls is not None:
                # Batch x Length x Channel
                logits_cls = self.logits_cls_softmax(logits_cls)
                # Batch x Length
                logits_cls = torch.argmax(logits_cls, 2)
            else:
                # Batch x Channels x Length x Length
                logits = self.logits_sigmoid(logits)

        if self.num_cls is not None:
            return logits, logits_cls
        else:
            return logits


class DIST(BasicDIST):
    """
    MAIN DIST FOR DIMENSIONLESS INSTANCE SEGMENTATION TRANSFORMER

    This transformer taking into the account the positional encoding of each
    coordinate point and attend them with patch image to which this coordinate
    is corresponding. This attention is aiming to training the transformer in
    outputting a graph from which point cloud can be segmented.


    Returns:
        torch.Tensor: DIST prediction after sigmoid (prediction) or last
            linear layer (training).
    """

    def __init__(self,
                 **kwargs):
        super(DIST, self).__init__(**kwargs)


class C_DIST(BasicDIST):
    """
    MAIN DIST FOR CLASSIFYING DIMENSIONLESS INSTANCE SEGMENTATION TRANSFORMER

    This transformer taking into the account the positional encoding of each
    coordinate point and attend them with patch image to which this coordinate
    is corresponding. This attention is aiming to training the transformer in
    outputting a graph from which point cloud can be segmented.

    Returns:
        torch.Tensor: DIST prediction as well as DIST class prediction, after
            sigmoid (prediction) or last linear layer (training).
    """

    def __init__(self,
                 **kwargs):
        super(C_DIST, self).__init__(**kwargs)
        assert self.num_cls is not None

        self.decoder_cls = nn.Linear(in_features=self.edge_dim,
                                     out_features=self.num_cls)

        if self.predict:
            self.logits_sigmoid = nn.Sigmoid()
        self.logits_cls_softmax = nn.Softmax(dim=2)
