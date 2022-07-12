from typing import Optional

import torch
from tardis.dist_pytorch.transformer.transformers import ComparisonLayer, GeluFeedForward, \
    PairBiasSelfAttention, SelfAttention2D, TriangularEdgeUpdate, QuadraticEdgeUpdate
from torch import nn


class GraphFormerStack(nn.Module):
    """
    WRAPPER FOR GRAPHFORMER LAYER

    This wrapper define number of layer for the graphformer.

    Args:
        node_dim: Number of input dimensions for node features.
        pairs_dim: Number of input dimension for pairs features.
        num_layers: Number of GraphFormer layers. Min. 1.
        ff_factor: Feed forward factor.
        num_heads: Number of heads in multi head attention.
    """

    def __init__(self,
                 node_dim: int,
                 pairs_dim: int,
                 num_layers=1,
                 dropout=0,
                 ff_factor=4,
                 num_heads=8,
                 structure='full'):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = GraphFormerLayer(pairs_dim=pairs_dim,
                                     node_dim=node_dim,
                                     dropout=dropout,
                                     ff_factor=ff_factor,
                                     num_heads=num_heads,
                                     structure=structure)
            self.layers.append(layer)

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

    def forward(self,
                z: torch.Tensor,
                x: Optional[torch.Tensor] = None,
                src_mask=None,
                src_key_padding_mask=None):
        for layer in self.layers:
            x, z = layer(h_pairs=x,
                         h_nodes=z,
                         src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask)
        return x, z


class GraphFormerLayer(nn.Module):
    """
    Main GraphFormer layer

    GraphFormerLayer takes an embedded input and perform the paired bias
    self attention (modified multi head attention), followed by GeLu feed
    forward normalization to update node embedded information. Then update from
    the GeLu is summed with edge feature map. As an output, GraphFormer output
    attention vector for given input that encoding attention between nodes and
    pairs (edges).

    Args:
        pairs_dim: Output feature for pairs and nodes representation
        node_dim: Input feature for pairs and nodes representation
        dropout: Dropout rate
        ff_factor: Feed forward factor used for GeLuFFN
        num_heads: Number of heads in self-attention
        structure: Structure of layer ['full', 'full_af', 'self_attn', 'triang', 'dualtraing', 'quad']
    """

    def __init__(self,
                 pairs_dim: int,
                 node_dim: Optional[int] = None,
                 dropout=0,
                 ff_factor=4,
                 num_heads=8,
                 structure='full'):
        super().__init__()
        self.pairs_dim = pairs_dim
        self.node_dim = node_dim
        self.structure = structure

        if node_dim is not None:
            self.input_attn = PairBiasSelfAttention(embed_dim=node_dim,
                                                    pairs_dim=pairs_dim,
                                                    num_heads=num_heads)
            self.input_ffn = GeluFeedForward(input_dim=node_dim,
                                             ff_dim=node_dim * ff_factor)
            self.pair_update = ComparisonLayer(input_dim=node_dim,
                                               output_dim=pairs_dim,
                                               channel_dim=pairs_dim)

        if self.structure in ['full', 'full_af', 'self_attn']:
            self.row_attention = SelfAttention2D(embed_dim=pairs_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout,
                                                 axis=1)
            self.col_attention = SelfAttention2D(embed_dim=pairs_dim,
                                                 num_heads=num_heads,
                                                 dropout=dropout,
                                                 axis=0)

        if self.structure in ['full', 'full_af', 'triang']:
            self.row_update = TriangularEdgeUpdate(input_dim=pairs_dim,
                                                   channel_dim=32,
                                                   axis=1)
            self.col_update = TriangularEdgeUpdate(input_dim=pairs_dim,
                                                   channel_dim=32,
                                                   axis=0)

        if self.structure == 'quad':
            self.row_update = QuadraticEdgeUpdate(input_dim=pairs_dim,
                                                  channel_dim=32,
                                                  axis=1)
            self.col_update = QuadraticEdgeUpdate(input_dim=pairs_dim,
                                                  channel_dim=32,
                                                  axis=0)
        if self.structure == 'dualtriang':
            self.row_update_1 = TriangularEdgeUpdate(input_dim=pairs_dim,
                                                  channel_dim=32,
                                                  axis=1)
            self.col_update_1 = TriangularEdgeUpdate(input_dim=pairs_dim,
                                                  channel_dim=32,
                                                  axis=0)

            self.row_update_2 = TriangularEdgeUpdate(input_dim=pairs_dim,
                                                  channel_dim=32,
                                                  axis=1)
            self.col_update_2 = TriangularEdgeUpdate(input_dim=pairs_dim,
                                                  channel_dim=32,
                                                  axis=0)
        self.pair_ffn = GeluFeedForward(input_dim=pairs_dim,
                                        ff_dim=pairs_dim * ff_factor)

    def update_nodes(self,
                     h_nodes: torch.Tensor,
                     h_pairs: Optional[torch.Tensor] = None,
                     src_mask=None,
                     src_key_padding_mask=None):
        """
        Transformer on the input weighted by the pair representations

        Input:
            h_nodes -> Batch x Length x Length x Channels
            h_pairs -> Length x Batch x Channels

        Output:
            h_pairs -> Length x Batch x Channels
        """
        h_pairs = h_pairs + self.input_attn(query=h_pairs,
                                            pairs=h_nodes,
                                            attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask)
        h_pairs = h_pairs + self.input_ffn(x=h_pairs)

        return h_pairs

    def update_edges(self,
                     h_nodes: torch.Tensor,
                     h_pairs: Optional[torch.Tensor] = None,
                     src_key_padding_mask=None):
        """
        Update the edge representations based on nodes

        Input:
            h_nodes -> Batch x Length x Length x Channels
            h_pairs -> Length x Batch x Channels

        Output:
            h_nodes -> Batch x Length x Length x Channels
        """
        if self.node_dim is not None and h_pairs is not None:
            h_nodes = h_nodes + self.pair_update(x=h_pairs)

        mask = None
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(2) | src_key_padding_mask.unsqueeze(1)

        if self.structure == 'full':
            h_nodes = h_nodes + \
                self.row_attention(x=h_nodes, padding_mask=mask) + self.col_attention(x=h_nodes, padding_mask=mask) + \
                self.row_update(z=h_nodes, mask=mask) + self.col_update(z=h_nodes, mask=mask)
        elif self.structure == 'full_af':
            h_nodes = h_nodes + self.row_attention(x=h_nodes, padding_mask=mask)
            h_nodes = h_nodes + self.col_attention(x=h_nodes, padding_mask=mask)
            h_nodes = h_nodes + self.row_update(z=h_nodes, mask=mask)
            h_nodes = h_nodes + self.col_update(z=h_nodes, mask=mask)
        elif self.structure == 'self_attn':
            h_nodes = h_nodes + self.row_attention(x=h_nodes, padding_mask=mask) + self.col_attention(x=h_nodes, padding_mask=mask)
        elif self.structure in ['triang', 'quad']:
            h_nodes = h_nodes + self.row_update(z=h_nodes, mask=mask) + self.col_update(z=h_nodes, mask=mask)
        elif self.structure == 'dualtriang':
            h_nodes = h_nodes + self.row_update_1(z=h_nodes, mask=mask) + self.col_update_1(z=h_nodes, mask=mask)
            h_nodes = h_nodes + self.row_update_2(z=h_nodes, mask=mask) + self.col_update_2(z=h_nodes, mask=mask)

        return h_nodes + self.pair_ffn(x=h_nodes)

    def forward(self,
                h_nodes: torch.Tensor,
                h_pairs: Optional[torch.Tensor] = None,
                src_mask=None,
                src_key_padding_mask=None):
        if self.node_dim is not None and h_pairs is not None:
            h_pairs = self.update_nodes(h_pairs=h_pairs,
                                        h_nodes=h_nodes,
                                        src_mask=src_mask,
                                        src_key_padding_mask=src_key_padding_mask)
        else:
            h_pairs = None

        h_nodes = self.update_edges(h_pairs=h_pairs,
                                    h_nodes=h_nodes,
                                    src_key_padding_mask=src_key_padding_mask)
        return h_pairs, h_nodes
