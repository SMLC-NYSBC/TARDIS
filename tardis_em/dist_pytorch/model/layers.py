#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################

from typing import Optional, Tuple

import torch
from torch import nn

from tardis_em.dist_pytorch.model.modules import (
    ComparisonLayer,
    GeluFeedForward,
    PairBiasSelfAttention,
    QuadraticEdgeUpdate,
    SelfAttention2D,
    TriangularEdgeUpdate,
)


class DistStack(nn.Module):
    """
    WRAPPER FOR DIST LAYER

    This wrapper defines a number of layer for the DIST.

    Args:
        node_dim (int): Number of input dimensions for node features.
        pairs_dim (int, optional): Number of input dimensions for pairs features.
        num_layers (int): Number of GraphFormer layers. Min. 1.
        dropout (float): Dropout rate.
        ff_factor (int): Feed forward factor.
        num_heads: Number of heads in multi-head attention.
        structure (str): Define DIST structure.
    """

    def __init__(
        self,
        pairs_dim: int,
        node_dim: Optional[int] = None,
        num_layers=1,
        dropout=0,
        ff_factor=4,
        num_heads=8,
        structure="full",
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = DistLayer(
                pairs_dim=pairs_dim,
                node_dim=node_dim,
                dropout=dropout,
                ff_factor=ff_factor,
                num_heads=num_heads,
                structure=structure,
            )
            self.layers.append(layer)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, i: int):
        return self.layers[i]

    def forward(
        self,
        edge_features: torch.Tensor,
        node_features: Optional[torch.Tensor] = None,
        src_mask=None,
        src_key_padding_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward throw individual DIST layer.

        Args:
            edge_features (torch.Tensor): Edge features as a tensor of shape
                [Batch x Length x Length x Channels].
            node_features (torch.Tensor, optional): Optional node features as a
                tensor of shape [Batch x Length x Channels].
            src_mask (torch.Tensor, optional): Optional source mask for masking
                over batches.
            src_key_padding_mask (torch.Tensor, optional): Optional mask use for
                feature padding.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Updated graph representation.
        """
        for layer in self.layers:
            node_features, edge_features = layer(
                h_pairs=edge_features,
                h_nodes=node_features,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        return node_features, edge_features


class DistLayer(nn.Module):
    """
    MAIN DIST LAYER

    DistLayer takes an embedded input and performs the paired bias
    self-attention (modified multi-head attention), followed by GeLu feed-forward
    normalization to update node-embedded information. Then update from
    the GeLu is summed with the edge feature map. As an output, DIST outputs
    an attention vector for given input that encodes attention between nodes and
    pairs (edges).

    Args:
        pairs_dim (int): Output feature for pairs and nodes representation.
        node_dim (int): Input feature for pairs and nodes representation.
        dropout (float): Dropout rate.
        ff_factor (int): Feedforward factor used for GeLuFFN.
        num_heads (int): Number of heads in self-attention
        structure (str): Structure of layer ['full', 'full_af', 'self_attn',
            'triang', 'dualtriang', 'quad'].
    """

    def __init__(
        self,
        pairs_dim: int,
        node_dim: Optional[int] = None,
        dropout=0,
        ff_factor=4,
        num_heads=8,
        structure="full",
    ):
        super().__init__()
        self.pairs_dim = pairs_dim
        self.channel_dim = int(pairs_dim / 4)
        if self.channel_dim <= 0:
            self.channel_dim = 1
        self.node_dim = node_dim
        self.structure = structure
        assert self.structure in [
            "full",
            "full_af",
            "self_attn",
            "triang",
            "quad",
            "dualtriang",
        ]

        # Node optional features update
        if node_dim is not None:
            if node_dim > 0:
                self.input_attn = PairBiasSelfAttention(
                    embed_dim=node_dim, pairs_dim=pairs_dim, num_heads=num_heads
                )
                self.input_ffn = GeluFeedForward(
                    input_dim=node_dim, ff_dim=node_dim * ff_factor
                )
                self.pair_update = ComparisonLayer(
                    input_dim=node_dim, output_dim=pairs_dim, channel_dim=pairs_dim
                )

        # Edge optional MHA update
        if self.structure in ["full", "full_af", "self_attn"]:
            self.row_attention = SelfAttention2D(
                embed_dim=pairs_dim, num_heads=num_heads, dropout=dropout, axis=1
            )
            self.col_attention = SelfAttention2D(
                embed_dim=pairs_dim, num_heads=num_heads, dropout=dropout, axis=0
            )

        # Edge triangular update
        if self.structure in ["full", "full_af", "triang"]:
            self.row_update = TriangularEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim
            )
            self.col_update = TriangularEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim, axis=0
            )

        # Edge Optional Quadratic
        if self.structure == "quad":
            self.row_update = QuadraticEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim
            )
            self.col_update = QuadraticEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim, axis=0
            )

        # Edge Optional dual-triang update
        if self.structure == "dualtriang":
            self.row_update_1 = TriangularEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim
            )
            self.col_update_1 = TriangularEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim, axis=0
            )

            self.row_update_2 = TriangularEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim
            )
            self.col_update_2 = TriangularEdgeUpdate(
                input_dim=pairs_dim, channel_dim=self.channel_dim, axis=0
            )

        # Edge GeLu FFN normalization layer
        self.pair_ffn = GeluFeedForward(
            input_dim=pairs_dim, ff_dim=pairs_dim * ff_factor
        )

    def update_nodes(
        self,
        h_pairs: torch.Tensor,
        h_nodes: Optional[torch.Tensor] = None,
        src_mask=None,
        src_key_padding_mask=None,
    ) -> torch.Tensor:
        """
        Transformer on the input weighted by the pair representations

        Input:
            h_paris -> Batch x Length x Length x Channels
            h_nodes -> Length x Batch x Channels

        Output:
            h_nodes -> Length x Batch x Channels

        Args:
            h_pairs (torch.Tensor): Edge features.
            h_nodes (torch.Tensor): Node features.
            src_mask (torch.Tensor): Attention mask used for mask over batch.
            src_key_padding_mask (torch.Tensor): Attention key padding mask.

        Returns:
             torch.Tensor: Updated node features.
        """
        h_nodes = h_nodes + self.input_attn(
            query=h_nodes,
            pairs=h_pairs,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        h_nodes = h_nodes + self.input_ffn(x=h_nodes)

        return h_nodes

    def update_edges(
        self,
        h_pairs: torch.Tensor,
        h_nodes: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask=None,
    ) -> torch.Tensor:
        """
        Update the edge representations based on nodes.

        Input:
            h_pairs -> Batch x Length x Length x Channels
            h_nodes -> Length x Batch x Channels

        Output:
            h_pairs -> Batch x Length x Length x Channels

        Args:
            h_pairs: Edge features.
            h_nodes: Node features.
            mask: Attention mask.
            src_key_padding_mask: Attention key padding mask.

        Returns:
            torch.Tensor: Updated edge features.
        """
        # Convert node features to edge shape
        if h_nodes is not None:
            h_pairs = h_pairs + self.pair_update(x=h_nodes)

        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(2) | src_key_padding_mask.unsqueeze(1)

        # Update edge features
        if self.structure == "full":
            h_pairs = (
                h_pairs
                + self.row_attention(x=h_pairs, padding_mask=mask)
                + self.col_attention(x=h_pairs, padding_mask=mask)
                + self.row_update(z=h_pairs, mask=mask)
                + self.col_update(z=h_pairs, mask=mask)
            )
        elif self.structure == "full_af":
            h_pairs = h_pairs + self.row_attention(x=h_pairs, padding_mask=mask)
            h_pairs = h_pairs + self.col_attention(x=h_pairs, padding_mask=mask)
            h_pairs = h_pairs + self.row_update(z=h_pairs, mask=mask)
            h_pairs = h_pairs + self.col_update(z=h_pairs, mask=mask)
        elif self.structure == "self_attn":
            h_pairs = (
                h_pairs
                + self.row_attention(x=h_pairs, padding_mask=mask)
                + self.col_attention(x=h_pairs, padding_mask=mask)
            )
        elif self.structure in ["triang", "quad"]:
            h_pairs = (
                h_pairs
                + self.row_update(z=h_pairs, mask=mask)
                + self.col_update(z=h_pairs, mask=mask)
            )
        elif self.structure == "dualtriang":
            h_pairs = (
                h_pairs
                + self.row_update_1(z=h_pairs, mask=mask)
                + self.col_update_1(z=h_pairs, mask=mask)
            )
            h_pairs = (
                h_pairs
                + self.row_update_2(z=h_pairs, mask=mask)
                + self.col_update_2(z=h_pairs, mask=mask)
            )

        return h_pairs + self.pair_ffn(x=h_pairs)

    def forward(
        self,
        h_pairs: torch.Tensor,
        h_nodes: Optional[torch.Tensor] = None,
        src_mask=None,
        src_key_padding_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapped forward throw all DIST layers.

        Args:
            h_pairs: Pairs representation.
            h_nodes: Node feature representation.
            src_mask: Optional attention mask.
            src_key_padding_mask: Optional padding mask for attention.

        Returns:
            Tuple[torch,Tensor, torch.Tensor]:
        """
        # Update node features and convert to edge shape
        if h_nodes is not None:
            h_nodes = self.update_nodes(
                h_pairs=h_pairs,
                h_nodes=h_nodes,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        # Update edge features
        h_pairs = self.update_edges(
            h_pairs=h_pairs, h_nodes=h_nodes, src_key_padding_mask=src_key_padding_mask
        )
        return h_nodes, h_pairs
