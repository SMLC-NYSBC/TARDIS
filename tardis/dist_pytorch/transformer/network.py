from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tardis.dist_pytorch.transformer.layers import GraphFormerStack
from tardis.dist_pytorch.transformer.modules import NodeEmbedding, EdgeEmbedding


class DIST(nn.Module):
    """
    MAIN CLASS OF SPATIAL-INSTANCE-SEGMENTATION TRANSFORMER

    This transformer taking into the account the positional encoding of each
    coordinate point and attend them with patch image to which this coordinate
    is corresponding. This attention is aiming to training the transformer in
    outputting a graph from which point cloud can be segmented.

    Args:
        n_out: Number of channels in the output layer.
        node_input: Length of the flattened image file.
        node_dim: In features of image for linear transformation.
        edge_dim: In feature of coord for linear transformation.
        num_layers: Number of graphformer layers to initialize.
        num_heads: Number of heads for MHA.
        coord_embed_sigma: Sigma value used to embed coordinate distance features.
        dropout_rate: Dropout factor used in MHA dropout layer
        structure: DIST network structure (full, traing, dualtriang, quad, attn)
        dist_embed: If True build dist embedding on single sigma.
        predict: If True sigmoid output

        coords: Coordinates input of a shape Batch x Channels x Length
        node_features: Image patch input of a shape Batch x Length x Dimensions
        padding_mask: Mask for additionally padding of input data
    """

    def __init__(self,
                 n_out=1,
                 node_input=0,
                 node_dim=None,
                 edge_dim=128,
                 num_layers=6,
                 num_heads=8,
                 coord_embed_sigma: Optional[tuple] = 1.0,
                 dropout_rate=0,
                 structure='full',
                 predict=False):
        super().__init__()

        self.node_input = node_input
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.predict = predict

        if node_input > 0:
            self.node_embed = NodeEmbedding(n_in=node_input,
                                            n_out=node_dim)

        self.coord_embed = EdgeEmbedding(n_out=edge_dim,
                                         sigma=coord_embed_sigma)

        self.layers = GraphFormerStack(node_dim=node_dim,
                                       pairs_dim=edge_dim,
                                       dropout=dropout_rate,
                                       num_layers=num_layers,
                                       num_heads=num_heads,
                                       structure=structure)

        self.decoder = nn.Linear(in_features=edge_dim,
                                 out_features=n_out)

        if self.predict:
            self.logits_sigmoid = nn.Sigmoid()

    def embed_input(self,
                    coords: torch.Tensor,
                    node_features: Optional[torch.Tensor] = None):
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
        """ Check if image patches exist """
        x, z = self.embed_input(coords=coords,
                                node_features=node_features)

        if x is not None:
            """ Length x Batch x Embedded_Dim """
            x = x.transpose(0, 1)

        """ Encode throughout the transformer layers """
        _, z = self.layers(x=x,
                           z=z)

        """ Predict the graph edges """
        z = z + z.transpose(1, 2)
        logits = self.decoder(z)  # symmetries z

        if self.predict:
            return self.logits_sigmoid(logits.permute(0, 3, 1, 2))
        else:
            return logits.permute(0, 3, 1, 2)


class C_DIST(nn.Module):
    """
    MAIN CLASS OF SPATIAL-INSTANCE-SEGMENTATION TRANSFORMER

    This transformer taking into the account the positional encoding of each
    coordinate point and attend them with patch image to which this coordinate
    is corresponding. This attention is aiming to training the transformer in
    outputting a graph from which point cloud can be segmented.

    Args:
        n_out: Number of channels in the output layer.
        node_input: Length of the flattened image file.
        node_dim: In features of image for linear transformation.
        edge_dim: In feature of coord for linear transformation.
        num_layers: Number of graphformer layers to initialize.
        num_heads: Number of heads for MHA.
        coord_embed_sigma: Sigma value used to embed coordinate distance features.
        dropout_rate: Dropout factor used in MHA dropout layer
        structure: DIST network structure (full, traing, dualtriang, quad, attn)
        dist_embed: If True build dist embedding on single sigma.
        predict: If True sigmoid output

        coords: Coordinates input of a shape Batch x Channels x Length
        node_features: Image patch input of a shape Batch x Length x Dimensions
        padding_mask: Mask for additionally padding of input data
    """

    def __init__(self,
                 n_out=1,
                 node_input=0,
                 node_dim=64,
                 edge_dim=128,
                 num_layers=6,
                 num_heads=8,
                 num_cls=200,
                 coord_embed_sigma: Optional[tuple] = 1.0,
                 dropout_rate=0,
                 structure='full',
                 predict=False):
        super().__init__()

        self.edge_dim = edge_dim
        self.predict = predict

        if node_input > 0:
            self.node_embed = NodeEmbedding(n_in=node_input,
                                            n_out=node_dim)

        self.coord_embed = EdgeEmbedding(n_out=edge_dim,
                                         sigma=coord_embed_sigma)

        self.layers = GraphFormerStack(node_dim=None,
                                       pairs_dim=edge_dim,
                                       dropout=dropout_rate,
                                       num_layers=num_layers,
                                       num_heads=num_heads,
                                       structure=structure)

        self.decoder = nn.Linear(in_features=edge_dim,
                                 out_features=n_out)
        self.decoder_cls = nn.Linear(in_features=edge_dim,
                                     out_features=num_cls)

        if self.predict:
            self.logits_sigmoid = nn.Sigmoid()
        self.logits_cls_softmax = nn.Softmax(dim=2)

    def embed_input(self,
                    coords: torch.Tensor,
                    node_features: Optional[torch.Tensor] = None):
        if node_features is not None:
            """ Batch x Length x Embedded_Dim """
            x = self.node_embed(input=node_features)
        else:
            x = None

        """ Batch x Length x Length x Channels """
        z = self.coord_embed(x=coords)
        return x, z

    def forward(self,
                coords: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None):
        x, z = self.embed_input(coords=coords,
                                node_features=None)

        if x is not None:
            """ Length x Batch x Embedded_Dim """
            x = x.transpose(0, 1)

        """ Encode throughout the transformer layers """
        _, z = self.layers(x=x,
                           z=z,
                           src_key_padding_mask=padding_mask)

        """ Predict the graph edges """
        logits = self.decoder(z + z.transpose(1, 2))  # symmetries z
        logits = logits.permute(0, 3, 1, 2)

        diag = np.arange(logits.shape[2])

        # Batch x Length x Channels
        logits_cls = self.decoder_cls(z)[:, diag, diag, :]
        logits_cls = self.logits_cls_softmax(logits_cls)  # Batch x Length x Channel

        if self.predict:
            logits = self.logits_sigmoid(logits)  # Batch x Channels x Length x Length
            logits_cls = torch.argmax(logits_cls, 2)  # Batch x Length

        return logits, logits_cls
