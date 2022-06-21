from typing import Optional

import torch
import torch.nn as nn
from tardis.dist_pytorch.transformer.layers import GraphFormerStack
from tardis.dist_pytorch.transformer.modules import DistEmbedding


class CloudToGraph(nn.Module):
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

        coords: Coordinates input of a shape Batch x Channels x Length
        node_features: Image patch input of a shape Batch x Length x Dimensions
        padding_mask: Mask for additionall padding of input data
    """

    def __init__(self,
                 n_out=1,
                 node_input=None,
                 node_dim=256,
                 edge_dim=128,
                 num_layers=6,
                 num_heads=8,
                 coord_embed_sigma=16,
                 dropout_rate=0,
                 structure='full',
                 dist_embed=True,
                 predict=False):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.predict = predict

        if node_input is not None:
            self.node_embed = nn.Linear(in_features=node_input,
                                        out_features=node_dim)

        self.coord_embed = DistEmbedding(n_out=edge_dim,
                                         sigma=coord_embed_sigma,
                                         dist=dist_embed)

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
        if hasattr(self, 'node_embed') and node_features is not None:
            """ Batch x Length x Embedded_Dim """
            x = self.node_embed(input=node_features)
        else:
            x = None

        """ Batch x Length x Length x Channels """
        z = self.coord_embed(x=coords)
        return x, z

    def forward(self,
                coords: torch.Tensor,
                node_features: Optional[torch.Tensor] = None,
                padding_mask: Optional[torch.Tensor] = None):
        """ Check if image patches exist """
        if node_features is not None and len(node_features.shape) != 3:
            node_features = None

        x, z = self.embed_input(coords=coords,
                                node_features=node_features)
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

        if self.predict:
            logits = self.logits_sigmoid(logits)

        return logits
