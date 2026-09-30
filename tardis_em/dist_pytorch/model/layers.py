#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2025                                            #
#######################################################################

import logging
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

logger = logging.getLogger(__name__)


class DistStack(nn.Module):
    """
    A neural network module for applying a stack of DistLayer layers to process
    edge and optional node features, typically for graph-based tasks.

    The DistStack class is a part of a Transformer-like architecture designed
    for graphs, where the stack is composed of multiple DistLayer layers. It
    provides an easy way to apply the stack sequentially on graph-related data,
    with the ability to handle optional node features and customizable edge
    masks for input feature attention.
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
        """
        Initializes a sequence of distributed layers for neural network processing.
        This class defines a module that creates and stores a series of layers, where
        each layer encapsulates various customizable parameters such as input dimensions,
        dropout rate, number of heads, and structural configuration.

        :param pairs_dim: The dimensional size of the pair-wise inputs for each layer.
        :param node_dim: Optional. The dimensional size of node-wise inputs for each layer. If not provided, defaults to `None`.
        :param num_layers: The total number of layers to instantiate. Defaults to `1`.
        :param dropout: Dropout probability for regularization within each layer. Defaults to `0`.
        :param ff_factor: Factor for feed-forward network scaling inside each layer. Defaults to `4`.
        :param num_heads: Number of attention heads to be used in each layer. Defaults to `8`.
        :param structure: Specifies the type/structure of the interaction within layers. Defaults to `"full"`.
        """
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
        inplace=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes input edge features and optionally node features through multiple layers, applying
        transformations to generate updated tensors for nodes and edges.

        The forward method iterates through all available layers, transforming the input features.
        Each layer processes the given feature sets (nodes and edges), along with optional masks
        (src_mask and src_key_padding_mask) as part of the computation. The output is updated node
        features and edge features after the transformations.

        :param edge_features: Input edge features as a tensor.
        :param node_features: Optional input node features, provided as a tensor.
                              Defaults to None.
        :param src_mask: Optional source mask for attention-based computations.
        :param src_key_padding_mask: Optional key padding mask for attention-based computations.
        :param inplace: If True, `edge_features` may be overwritten when gradients
                        are disabled. Saves one copy of the edge features.
        :return: A tuple consisting of two tensors:
                 - Updated node features
                 - Updated edge features
        """
        for i, layer in enumerate(self.layers):
            node_features, edge_features = layer(
                h_pairs=edge_features,
                h_nodes=node_features,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                # After the first layer, edge_features is owned by this stack
                inplace=inplace or i > 0,
            )

        return node_features, edge_features


class DistLayer(nn.Module):
    """
    DistLayer class is designed for hierarchical processing of node and pair
    representations using multi-head attention and feed-forward mechanisms.
    The class supports various structures for interaction layers such as
    triangular, quadratic, dual triangular updates, or full attention-based
    architectures. It extends PyTorch's nn.Module and incorporates mechanisms
    to handle input features, as well as dropout for regularization.

    DistLayer allows for versatile interaction between node and pair features
    through specific update routines that depend on the chosen structure.
    """

    # Pair positions (rows x columns) processed at once when gradients are disabled
    inference_chunk_size = 2**15

    def __init__(
        self,
        pairs_dim: int,
        node_dim: Optional[int] = None,
        dropout=0,
        ff_factor=4,
        num_heads=8,
        structure="full",
    ):
        """
        Initializes the class with parameters for edge and node feature updates, defining
        transformations and attention mechanisms based on a specified structure type. The
        structure defines how the relationships between pairs of data points and optional node
        features are updated and refined, ensuring flexibility in design and functionality.

        :param pairs_dim: Dimensionality of the pair features.
        :type pairs_dim: int
        :param node_dim: Optional dimensionality of node features. If None, node features are not used.
        :type node_dim: int or None
        :param dropout: Dropout probability applied to attention layers.
        :type dropout: float
        :param ff_factor: Feed-forward expansion factor. Defines the scaling for intermediate
                          linear dimensions in feed-forward layers.
        :type ff_factor: int
        :param num_heads: Number of attention heads for multi-head attention.
        :type num_heads: int
        :param structure: The type of structural update to apply. Must be one of the following:
                          "full", "full_af", "self_attn", "triang", "quad", or "dualtriang".
        :type structure: str
        """
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
        # row_attention attends over i for each j, col_attention over j for each i
        if self.structure in ["full", "full_af", "self_attn"]:
            self.row_attention = SelfAttention2D(
                embed_dim=pairs_dim, num_heads=num_heads, dropout=dropout, axis=0
            )
            self.col_attention = SelfAttention2D(
                embed_dim=pairs_dim, num_heads=num_heads, dropout=dropout, axis=1
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
        Updates the node representations based on pair embeddings and self-attention
        mechanism. This function combines the provided node embeddings and pair
        embeddings through an attention mechanism and applies a feed-forward network
        for further transformation. The updated node representation is returned.

        :param h_pairs: Pairwise embeddings. A tensor that provides information about
            pair dependencies between nodes.
        :param h_nodes: Optional initial node embeddings. If provided, these will be
            updated using the attention mechanism and feed-forward network.
        :param src_mask: Attention mask used during the attention computation to
            indicate valid positions. This allows selective attention and prevents
            unwanted information flow.
        :param src_key_padding_mask: Key padding mask used to indicate valid and
            invalid tokens or nodes for each sample in a batch. Useful during
            variable-length sequence handling.
        :return: Updated node representations after attention computation and
            feed-forward network application.
        :rtype: torch.Tensor
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
        inplace=False,
    ) -> torch.Tensor:
        """
        Updates the edge features in a graph based on the chosen structural configuration
        and optionally includes node features or masking conditions. The method modifies
        the input edge features by applying a variety of attentions, feature updates,
        and feedforward transformations, depending on the structure type.

        :param h_pairs: Tensor containing initial edge features, of shape
            `(batch_size, num_nodes, num_nodes, feature_dim)`.
        :param h_nodes: Optional tensor containing node features, of shape
            `(batch_size, num_nodes, feature_dim)`. If provided, the function incorporates
            these features into the edge features during the update process.
        :param mask: Optional tensor of shape `(batch_size, num_nodes, num_nodes)`. Acts
            as an attention mask or structural constraint for the feature update process.
        :param src_key_padding_mask: Optional tensor of shape `(batch_size, num_nodes)`. If
            provided, it is used to generate a mask to ignore certain nodes by expanding it
            along the necessary dimensions.
        :param inplace: If True, `h_pairs` may be overwritten when gradients are disabled.
        :return: A tensor of the same shape as `h_pairs`, representing the updated edge
            features after applying the selected transformations.
        """
        if (
            not torch.is_grad_enabled()
            and mask is None
            and src_key_padding_mask is None
        ):
            return self._update_edges_no_grad(
                h_pairs=h_pairs, h_nodes=h_nodes, inplace=inplace
            )

        # Convert node features to edge shape
        if h_nodes is not None:
            h_pairs = h_pairs + self.pair_update(x=h_nodes)

        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.unsqueeze(2) | src_key_padding_mask.unsqueeze(1)

        h_pairs = self._update_structure(h_pairs=h_pairs, mask=mask)

        return h_pairs + self.pair_ffn(x=h_pairs)

    def _update_structure(
        self, h_pairs: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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

        return h_pairs

    def _update_edges_no_grad(
        self,
        h_pairs: torch.Tensor,
        h_nodes: Optional[torch.Tensor] = None,
        inplace=False,
    ) -> torch.Tensor:
        """
        Memory-efficient version of `update_edges` for inference without masks.

        Gives the same result as `update_edges`, but adds each residual update
        into `h_pairs` in place, one block of rows at a time. The large
        intermediates (up to 4 x pairs_dim channels in the feed-forward network)
        then exist only for one block instead of the full L x L grid. Triangular
        updates are split this way; the other structures only split the
        feed-forward network.

        This path does not support autograd because it writes in place.

        :param h_pairs: Edge features (batch_size, num_nodes, num_nodes, feature_dim).
        :param h_nodes: Optional node features.
        :param inplace: If True, `h_pairs` may be overwritten; otherwise it is copied
            before the first in-place update.
        :return: Updated edge features.
        """
        if h_nodes is not None:
            h_pairs = h_pairs + self.pair_update(x=h_nodes)
            inplace = True  # h_pairs is now a new tensor

        batch, length = h_pairs.shape[0], h_pairs.shape[1]
        rows = max(1, self.inference_chunk_size // (batch * length))

        if self.structure in ["triang", "dualtriang"]:
            if not inplace:
                h_pairs = h_pairs.clone()

            if self.structure == "triang":
                updates = [(self.row_update, self.col_update)]
            else:
                updates = [
                    (self.row_update_1, self.col_update_1),
                    (self.row_update_2, self.col_update_2),
                ]

            # h = h + row_update(h) + col_update(h), computed block by block.
            # Row block i of the update only needs row block i of h and the
            # triangular products k, which are computed before h is changed.
            for row_update, col_update in updates:
                row_k = row_update.contract(h_pairs, rows)
                col_k = col_update.contract(h_pairs, rows)

                for start in range(0, length, rows):
                    h_rows = h_pairs[:, start : start + rows]
                    row = row_update.forward_rows(h_rows, row_k, start)
                    col = col_update.forward_rows(h_rows, col_k, start)
                    h_rows += row
                    h_rows += col
                del row_k, col_k
        else:
            h_pairs = self._update_structure(h_pairs=h_pairs)

        for start in range(0, length, rows):
            h_rows = h_pairs[:, start : start + rows]
            h_rows += self.pair_ffn(x=h_rows)

        return h_pairs

    def forward(
        self,
        h_pairs: torch.Tensor,
        h_nodes: Optional[torch.Tensor] = None,
        src_mask=None,
        src_key_padding_mask=None,
        inplace=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes and updates node and edge features using provided input tensors.
        The function advances the transformation of node and edge features by applying
        update operations on input tensors, including optional masking of source inputs.

        :param h_pairs: Tensor representing the edge features in the graph.
        :param h_nodes: Tensor representing the node features in the graph. It can
            be optionally None, in which case no node updates will be performed.
        :param src_mask: Optional mask applied at the source level during node update.
        :param src_key_padding_mask: Optional mask to specify which elements should be
            ignored in the computation, typically used for padded sequences.
        :param inplace: If True, `h_pairs` may be overwritten when gradients are disabled.

        :return: Tuple of two tensors, the updated node features and the updated edge
            features in the graph.
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
            h_pairs=h_pairs,
            h_nodes=h_nodes,
            src_key_padding_mask=src_key_padding_mask,
            inplace=inplace,
        )
        return h_nodes, h_pairs
