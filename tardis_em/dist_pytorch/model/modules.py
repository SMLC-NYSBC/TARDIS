#######################################################################
#  TARDIS - Transformer And Rapid Dimensionless Instance Segmentation #
#                                                                     #
#  New York Structural Biology Center                                 #
#  Simons Machine Learning Center                                     #
#                                                                     #
#  Robert Kiewisz, Tristan Bepler                                     #
#  MIT License 2021 - 2024                                            #
#######################################################################
from math import sqrt
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairBiasSelfAttention(nn.Module):
    """
    Implements self-attention mechanism that incorporates pairwise features
    for multi-head attention. This class is designed for scenarios where
    attention bias is calculated based on edge features alongside the node
    embeddings. The module allows adjustable parameters including the number
    of heads, embedding dimensions, and initial scaling factors.

    The attention mechanism applies normalization and linear projections to
    both node and edge features, calculates attention weights, and uses those
    weights to compute weighted combinations of feature representations.
    """

    def __init__(
        self,
        embed_dim: int,
        pairs_dim: int,
        num_heads: int,
        init_scaling=1 / 1.4142135623730951,
    ):
        """
        Initializes the multi-head attention module with the specified embedding dimensions,
        number of attention heads, and scaling factors. The module creates the required
        linear layers for processing query, key, value embeddings, and additional
        pairs embeddings.

        :param embed_dim:
            The dimensionality of the embedding space for queries, keys, and values.
        :param pairs_dim:
            The dimensionality of the additional pairs embeddings used as input.
        :param num_heads:
            The number of attention heads used in the multi-head attention mechanism.
        :param init_scaling:
            The initial scaling factor for weights' initialization.
        """
        super().__init__()
        # Embedding setting
        self.embed_dim = self.kdim = self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        # Initializing scaling factor
        self.init_scaling = init_scaling

        # Embedding linear layers
        self.norm_input = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=False)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.pairs_dim = pairs_dim
        self.norm_pairs = nn.LayerNorm(pairs_dim)
        self.pairs_proj = nn.Linear(self.pairs_dim, self.num_heads, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initializes the parameters of the network layers using the Xavier uniform
        distribution and constants for better convergence during training.

        This method resets the weights and biases of specific layers in the
        network as per the defined initialization strategies. The Xavier uniform
        distribution is applied to certain weights, while constants are used for
        biases and selected weights. If a bias exists for the output projection
        layer, it is initialized to zero.

        :raises AttributeError: If any of the required layer attributes
            (e.g., q_proj, k_proj, v_proj, out_proj, pairs_proj) are not present
            in the current instance.
        :return: None
        """
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)

        nn.init.constant_(self.out_proj.weight, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.pairs_proj.weight, gain=self.init_scaling)

    def forward(
        self,
        query: torch.Tensor,
        pairs: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Computes the forward pass of a multi-head attention module with additional pairwise
        positional weighting. This function first processes the query tensor through normalization
        and projection, prepares the attention weights using pair and masked components, applies
        the attention and then combines attended outputs with pairwise positional contributions.
        It optionally provides attention weights for inference or analysis.

        This method supports key padding masks for excluding irrelevant tokens from attention calculations,
        as well as optional attention masks for customizing attention strength in multi-head
        contexts. The function includes the necessary transformations to prepare query,
        key, and value tensors for batched multi-head operations.

        :param query: The input tensor of shape (target length, batch size, embedding dimension).
        :param pairs: Pairwise positional tensor of shape (batch size, sequence length,
                      sequence length, number of heads).
        :param attn_mask: Optional mask tensor of shape (target length, source length) to
                          apply additional additive masking to the attention weights.
        :param key_padding_mask: Optional binary tensor of shape (batch size, source length)
                                 indicating padded positions.
        :param need_weights: Boolean flag to indicate whether to return the attention weights
                             alongside the output tensor.
        :param need_head_weights: Boolean flag to indicate whether individual head weights
                                  should be returned. Overrides `need_weights` when True.
        :return: The output tensor of shape (target length, batch size, embedding dimension).
                 If `need_weights` is True, also returns attention weights; the shape of weights
                 depends on the `need_head_weights` parameter.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        query = self.norm_input(query)
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        """
        This is part of a workaround to get around fork/join parallelism
        not supporting Optional types.
        """
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        pairs = self.norm_pairs(pairs)
        pair_weights = self.pairs_proj(pairs)

        # pair_weights are B x L x L x H, needs to be B x H x L x L
        pair_weights = pair_weights.permute(0, 3, 1, 2)
        attn_weights = (
            attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + pair_weights
        )
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)

            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)

            return attn, attn_weights
        return attn


class ComparisonLayer(nn.Module):
    """
    Defines the ComparisonLayer class, which is used to process and transform node
    features into a specific tensor shape. The class includes normalization and
    linear transformations for enhanced data manipulation.

    This layer takes as input node features and performs a series of transformations
    to generate a tensor that is compatible with specific downstream tasks in deep
    learning models. It leverages PyTorch's `nn.Module` for implementing
    customized neural network layers.
    """

    def __init__(self, input_dim: int, output_dim: int, channel_dim=128):
        """
        This class initializes a module that transforms input node features into edge-specific
        features. It includes a sequence of linear transformations and normalization layers,
        with explicit weight and bias initialization.

        :param input_dim: Specifies the dimension of the input node feature vector.
        :type input_dim: int
        :param output_dim: Specifies the desired output dimension of the edge feature vector.
        :type output_dim: int
        :param channel_dim: Specifies the intermediate channel dimension used in transformations.
        Defaults to 128 if not provided.
        :type channel_dim: int
        """
        super().__init__()
        # Linear layer for converting pairs node futures to edge shape.
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, channel_dim)
        self.linear2 = nn.Linear(input_dim, channel_dim)
        self.linear3 = nn.Linear(channel_dim, output_dim)
        self.linear4 = nn.Linear(channel_dim, output_dim, bias=False)

        # Parameter initialization
        nn.init.constant_(self.linear3.weight, 0.0)
        nn.init.constant_(self.linear3.bias, 0.0)

        nn.init.constant_(self.linear4.weight, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the computational graph built for the operation.
        The method applies specific transformations on the input tensor such as transposition,
        normalization, and linear transformations to generate the final output. The computation
        includes element-wise multiplication and subtraction of transformed tensors, followed
        by additional linear transformations.

        :param x: Input tensor with shape (Batch, Length, Feature_Dimensions)
        :type x: torch.Tensor
        :return: Transformed tensor with shape (Batch, Length, Length, Out_Channels)
        :rtype: torch.Tensor
        """
        x = x.transpose(0, 1)
        x = self.norm(x)
        a = self.linear1(x)
        b = self.linear2(x)

        """Batch x Length x Length x Out_Channels"""
        return self.linear3(a.unsqueeze(2) * b.unsqueeze(1)) + self.linear4(
            a.unsqueeze(2) - b.unsqueeze(1)
        )


class TriangularEdgeUpdate(nn.Module):
    """
    The TriangularEdgeUpdate class implements a neural network module that performs
    triangular edge updates for edge feature tensors. This is primarily designed for
    processing relational or structural data, where edge updates between nodes in a
    triangular relationship must be computed.

    The class takes input and processes it using linear layers, layer normalization,
    and gating mechanisms. It supports optional masking and performs updates
    based on a defined axis. The resulting features are computed via einsum operations,
    allowing for flexible interaction across specified dimensions.
    """

    def __init__(self, input_dim, channel_dim=128, axis=1):
        """
        Initializes the custom class with specified input dimensions, channel dimensions,
        and axis for performing linear operations and gating mechanisms.

        :param input_dim: Dimensionality of input features, used to define the shape for
            layers and transformations.
        :type input_dim: int
        :param channel_dim: Dimensionality of the intermediate channel, used for operations
            in the linear and gating mechanisms. Defaults to 128 if not provided.
        :type channel_dim: int, optional
        :param axis: Axis along which the operations are performed. Defaults to 1 if
            not explicitly specified.
        :type axis: int, optional
        """
        super().__init__()
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.init_scaling = 1 / sqrt(2)

        self.norm_input = nn.LayerNorm(input_dim)

        self.linear_a = nn.Linear(input_dim, channel_dim)
        self.gate_a = nn.Linear(input_dim, channel_dim)

        self.linear_b = nn.Linear(input_dim, channel_dim)
        self.gate_b = nn.Linear(input_dim, channel_dim)

        self.norm_o = nn.LayerNorm(channel_dim)
        self.gate_o = nn.Linear(input_dim, input_dim)
        self.linear_o = nn.Linear(channel_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resets the parameters of the network layers to their initial values.

        The method initializes the weights and biases of the linear layers `linear_a`,
        `linear_b`, and `linear_o` using specific initialization techniques. For
        `linear_a` and `linear_b`, the weights are initialized using Xavier uniform
        initialization, scaled by the `init_scaling` value, and their biases are set
        to a constant value of 0. For `linear_o`, both weights and biases are set
        to a constant value of 0.

        :rtype: None
        :return: The method does not return any value.
        """
        nn.init.xavier_uniform_(self.linear_a.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_a.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_b.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_b.bias, 0.0)

        nn.init.constant_(self.linear_o.weight, 0.0)
        nn.init.constant_(self.linear_o.bias, 0.0)

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Processes the input tensor `z` using gated mechanisms and performs tensor operations to produce the output.
        It applies normalization, gating, and optionally masks certain elements based on the provided mask tensor.
        The computation involves einsum operations and a final gating mechanism for the output tensor.

        :param z: Input tensor with dimensions suitable for processing by the forward method.
        :param mask: Optional tensor used to mask specific elements of the input tensor during processing.
            If provided, elements are masked where the mask tensor indicates.
        :return: Processed tensor after applying the normalization, gating, masking (if applicable),
            and tensor manipulation operations.
        """
        z = self.norm_input(z)

        a = torch.sigmoid(self.gate_a(z)) * self.linear_a(z)  # B x L x L x O
        b = torch.sigmoid(self.gate_b(z)) * self.linear_b(z)  # B x L x L x O

        if mask is not None:
            mask = mask.unsqueeze(3).expand(
                mask.size(0), mask.size(1), mask.size(2), a.size(3)
            )
            a = torch.where(mask, torch.zeros_like(a), a)
            b = torch.where(mask, torch.zeros_like(b), b)

        # i,j -> i,k j,k
        if self.axis == 1:
            k = torch.einsum("biko,bjko->bijo", a, b)
        else:
            k = torch.einsum("bkio,bkjo->bijo", a, b)

        return torch.sigmoid(self.gate_o(z)) * self.linear_o(self.norm_o(k))


class QuadraticEdgeUpdate(nn.Module):
    """
    A neural network module for quadratic edge updates with gated linear units and layer normalization.

    This module processes edge features through a series of transformations, including gated linear
    layer calculations, layer normalization, and tensor manipulation using the einsum operation. It
    supports masking for optional selective computation over input tensors and provides flexible
    dimensional configurations.

    The input is normalized first and then transformed via multiple linear and gating operations.
    The outputs are combined through einsum-based operations based on the configured axis, enabling
    contextual computations for edge features. The final results are subjected to normalization and
    linear transformations to produce the output tensor.
    """

    def __init__(self, input_dim, channel_dim=128, axis=1):
        """
        A class for implementing a neural network transformation with gated linear units
        (GLU) and layer normalization applied for dimensional transformations.

        This module performs operations through a multi-step pipeline, where inputs
        undergo normalization, linear transformation, gating mechanisms, and then
        final output layer adjustments. It leverages nn.LayerNorm and nn.Linear layers
        to process inputs with custom dimensions and scales. Parameters for dimensions
        and specific operational axes are configurable.

        The class encapsulates parameter initialization and ensures stable training
        through default scaling techniques.

        Attributes:
            input_dim: Dimensionality of input data.
            channel_dim: Number of channels used in intermediate computations.
            axis: The tensor axis over which transformations are computed.
            init_scaling: Scaling factor for parameter initialization.
            norm_input: LayerNorm operation applied to the input.
            linear_a, linear_b, linear_c, linear_d: Linear transformation layers.
            gate_a, gate_b, gate_c, gate_d: Gating layers to condition tensor values.
            norm_o: LayerNorm operation applied after intermediate processing.
            gate_o: Final gating layer for output adjustments.
            linear_o: Final linear transformation for generating output.

        :param input_dim: Dimensionality of the input tensor to the module.
        :type input_dim: int
        :param channel_dim: Number of channels for intermediate transformation, defaults to 128.
        :type channel_dim: int
        :param axis: Axis for dimensional transformation, defaults to 1.
        :type axis: int
        """
        super().__init__()
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.init_scaling = 1 / sqrt(2)

        self.norm_input = nn.LayerNorm(input_dim)

        self.linear_a = nn.Linear(input_dim, channel_dim)
        self.gate_a = nn.Linear(input_dim, channel_dim)

        self.linear_b = nn.Linear(input_dim, channel_dim)
        self.gate_b = nn.Linear(input_dim, channel_dim)

        self.linear_c = nn.Linear(input_dim, channel_dim)
        self.gate_c = nn.Linear(input_dim, channel_dim)

        self.linear_d = nn.Linear(input_dim, channel_dim)
        self.gate_d = nn.Linear(input_dim, channel_dim)

        self.norm_o = nn.LayerNorm(channel_dim)
        self.gate_o = nn.Linear(input_dim, input_dim)
        self.linear_o = nn.Linear(channel_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Resets the parameters of the module by initializing weights and biases for all
        linear layers. All weights are initialized using Xavier uniform initialization
        with the respective gain, and all biases are initialized to zero.

        :raises RuntimeError: If the module attributes `linear_a`, `linear_b`,
           `linear_c`, `linear_d`, or `linear_o` are not properly initialized
           as ``torch.nn.Linear`` objects before calling this method.
        :return: None
        """
        nn.init.xavier_uniform_(self.linear_a.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_a.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_b.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_b.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_c.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_c.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_d.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_d.bias, 0.0)

        nn.init.constant_(self.linear_o.weight, 0.0)
        nn.init.constant_(self.linear_o.bias, 0.0)

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes the forward pass of the layer by applying gated transformations to the input tensor,
        optionally considering an input mask. It applies a series of linear transformations and
        element-wise sigmoid activations to generate intermediate tensors, followed by a tensor contraction
        using Einstein summation notation for specific axes. The output is further transformed by gating
        and normalization operations.

        :param z: The input tensor with shape (B x L x D) where B is the batch size, L is the sequence length,
            and D is the input dimension.
        :param mask: An optional binary mask tensor with shape (B x L x L) where masked positions are indicated
            with `1` and unmasked positions with `0`. If provided, it will nullify specific computations in the output.
        :return: A tensor with shape (B x L x L x O), representing the transformed outputs after gating,
            linear transformation, normalization, and contraction operations.
        """
        z = self.norm_input(z)

        a = torch.sigmoid(self.gate_a(z)) * self.linear_a(z)  # B x L x L x O
        b = torch.sigmoid(self.gate_b(z)) * self.linear_b(z)  # B x L x L x O
        c = torch.sigmoid(self.gate_c(z)) * self.linear_c(z)  # B x L x L x O
        d = torch.sigmoid(self.gate_d(z)) * self.linear_d(z)  # B x L x L x O

        if mask is not None:
            mask = mask.unsqueeze(3).expand(
                mask.size(0), mask.size(1), mask.size(2), a.size(3)
            )
            a = torch.where(mask, torch.zeros_like(a), a)
            b = torch.where(mask, torch.zeros_like(b), b)
            c = torch.where(mask, torch.zeros_like(c), c)
            d = torch.where(mask, torch.zeros_like(d), d)

        # i,j -> i,k j,k, il, jl
        if self.axis == 1:
            k = torch.einsum("biko,bjko,bilo,bjlo->bijo", a, b, c, d)
        else:
            k = torch.einsum("bkio,bkjo,blio,bljo->bijo", a, b, c, d)

        return torch.sigmoid(self.gate_o(z)) * self.linear_o(self.norm_o(k))


class MultiHeadAttention(nn.Module):
    """
    Represents a Multi-Head Attention (MHA) mechanism that enables self-attention
    or cross-attention in neural networks, primarily used in transformers.
    MHA facilitates attention over multiple heads, allowing the model to focus
    on different parts of the sequence simultaneously. This module supports
    various options such as encoder-decoder attention, dropout, bias customization,
    and scaling initialization.

    Multi-Head Attention is a key building block for many Natural Language
    Processing (NLP) and computer vision tasks, enabling the model to capture
    contextual dependencies efficiently.

    The module expects inputs in the form of query, key, and value tensors and
    provides attention outputs for further processing in the neural network.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        init_scaling=1 / sqrt(2),
    ):
        """
        Initializes the multi-head attention layer. This layer is designed to process
        queries, keys, and values, providing flexibility for variable input sizes
        depending on the use case, whether it's self-attention, encoder-decoder
        attention, or general multi-head attention mechanisms. It configures key
        components such as projection layers, dropout, and optional biases for keys and
        values.

        :param embed_dim: Embedding dimension of the model.
        :type embed_dim: int

        :param num_heads: Number of attention heads.
        :type num_heads: int

        :param kdim: Key dimension. If None, defaults to the value of embed_dim.
        :type kdim: Optional[int]

        :param vdim: Value dimension. If None, defaults to the value of embed_dim.
        :type vdim: Optional[int]

        :param dropout: Dropout probability applied during attention.
        :type dropout: float

        :param bias: If True, includes a bias term in projection layers.
        :type bias: bool

        :param add_bias_kv: If True, adds learnable bias to keys and values.
        :type add_bias_kv: bool

        :param add_zero_attn: If True, adds an all-zero attention frame.
        :type add_zero_attn: bool

        :param self_attention: If True, configures layer for self-attention.
        :type self_attention: bool

        :param encoder_decoder_attention: If True, configures layer for encoder-decoder attention.
        :type encoder_decoder_attention: bool

        :param init_scaling: Scaling factor applied during initialization.
        :type init_scaling: float
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert (
            not self.self_attention or self.qkv_same_dim
        ), "Self-attention requires query, key and value to be of the same size"

        self.init_scaling = init_scaling
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def reset_parameters(self):
        """
        Initializes model parameters with specific initialization methods.

        This method resets the weights and biases of the key, value, query, and output
        projection layers in the neural network to ensure consistent training results
        and proper initialization of the model. It uses Xavier uniform initialization
        for the weights and constant initialization for specific components such as
        biases. If optional biases `bias_k` or `bias_v` exist, they are also reset to
        constant values.

        :raises TypeError: An exception is raised if the model components are not
                           properly initialized or invalid attribute references to occur.
        """
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.init_scaling)
        nn.init.constant_(self.out_proj.weight, 0.0)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.constant_(self.bias_k.bias, 0.0)
        if self.bias_v is not None:
            nn.init.constant_(self.bias_v.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Computes the forward pass through a multi-head attention mechanism with
        support for self-attention, encoder-decoder attention, and custom input
        masks. The function supports options to include or exclude weights,
        apply dropout, and handle special cases such as biases and padding.

        :param query: Tensor representing the input sequence to compute attention for,
            with dimensions (target length, batch size, embedding dimension).
        :param key: Optional tensor representing the key in the attention mechanism,
            with dimensions (source length, batch size, embedding dimension). If None,
            it assumes self-attention or other specialized attention types.
        :param value: Optional tensor representing the value in the attention mechanism,
            with dimensions (source length, batch size, embedding dimension). If None,
            it assumes self-attention or other specialized attention types.
        :param key_padding_mask: Optional boolean tensor used to specify padding on
            certain input positions, with dimensions (batch size, source length). Non-zero
            values denote positions to be masked.
        :param need_weights: Boolean flag indicating whether the function should return
            attention weights along with the computed output.
        :param attn_mask: Optional tensor representing a mask to restrict attention to
            specific positions, with dimensions (target length, source length). Typical
            for causal masking in transformer models.
        :param before_softmax: Boolean flag to determine if attention weights are returned
            before or after the softmax operation is applied.
        :param need_head_weights: Boolean flag indicating whether attention weights per
            head are needed (as opposed to aggregated weights across heads).
        :return: A tuple consisting of the attention output (tensor with dimensions
            (target length, batch size, embedding dimension)) and, if requested, the
            attention weights (as a tensor with details depending on the head weights
            selection). If weights are not requested, only the attention output is returned.

        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        assert k is not None
        src_len = k.size(1)

        """
        This is part of a workaround to get around fork/join parallelism
        not supporting Optional types.
        """
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            """don't attend to padding symbols"""
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            """
            When ONNX tracing a single decoder step (sequence length == 1)
            the transpose is a no-op copy before view, thus unnecessary.
            """
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if not need_head_weights:
                """average attention weights over heads"""
                attn_weights = attn_weights.mean(dim=1)

            return attn, attn_weights

        return attn


class SelfAttention2D(MultiHeadAttention):
    """
    Implements 2D self-attention mechanism.

    This class extends the MultiHeadAttention module to perform self-attention
    specifically over 2D edge features. It provides functionality to reshape
    the input features depending on the axis mode (rows or columns) and enables
    efficient computation of attention by considering memory constraints via
    batching.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, axis=None, dropout=0.0, max_size=4194304
    ):
        """
        Represents a 2D self-attention mechanism with configurable embedding dimensions,
        number of attention heads, and other parameters.

        This class is designed to extend the functionality of the
        multi-head self-attention mechanism, with additional control over specific
        axes and maximum size constraints.

        :param embed_dim: The dimensionality of the input embeddings.
        :type embed_dim: int
        :param num_heads: The number of attention heads.
        :type num_heads: int
        :param axis: The axis or axes along which attention is applied.
        :type axis: Optional or defined type of the axis
        :param dropout: The dropout rate applied during training.
        :type dropout: float
        :param max_size: The maximum size of the attention mechanism. Defaults to 4194304.
        :type max_size: int
        """
        super(SelfAttention2D, self).__init__(
            embed_dim, num_heads, dropout=dropout, self_attention=True
        )
        self.axis = axis
        self.max_size = max_size

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """
        Processes input tensor through a 2D self-attention mechanism and adjusts its shape
        based on the specified axis. Handles padding masks if provided, allowing optional
        batching for memory-efficient computation when the attention matrix size exceeds
        a specified maximum.

        :param x: Input tensor containing the features to be processed using 2D self-attention.
        :type x: torch.Tensor
        :param padding_mask: Optional mask used to ignore certain positions during the
            attention computation.
        :type padding_mask: torch.Tensor or None
        :return: Transformed tensor with the same spatial dimensions as the input but
            adjusted for the attention weights applied.
        :rtype: torch.Tensor
        """

        R, C, B, DIM = x.size()
        axis = self.axis
        if axis is None:
            x = x.view(R * C, B, DIM)
            if padding_mask is not None:
                padding_mask = padding_mask.view(B, R * C)
        else:
            assert axis == 0 or axis == 1

            """attend along the row dimension"""
            if axis == 0:
                x = x.view(R, C * B, DIM)
                if padding_mask is not None:
                    padding_mask = padding_mask.permute(2, 0, 1)
                    padding_mask = padding_mask.reshape(C * B, R)
                """attend along the col dimension"""
            else:
                x = x.transpose(0, 1)
                x = x.reshape(C, R * B, DIM)
                if padding_mask is not None:
                    padding_mask = padding_mask.permute(1, 0, 2)
                    padding_mask = padding_mask.reshape(R * B, C)

        if 0 < self.max_size < x.size(0) ** 2 * x.size(1):
            """
            Attention matrix size times batch size will exceed maximum
            allowable entries split into batches to make attention matrix RAM
            workable calculating attention over batches helps reduce RAM when
            N or M is large
            """
            batch_size = x.size(0) ** 2 // self.max_size
            if batch_size < 1:
                """might run out of RAM, but batch size can't be < 1"""
                batch_size = 1

            h = []
            for i in range(0, x.size(1), batch_size):
                xi = x[:, i : i + batch_size]
                mask = None
                if padding_mask is not None:
                    mask = padding_mask[i : i + batch_size]
                h.append(
                    super(SelfAttention2D, self).forward(xi, key_padding_mask=mask)
                )
            h = torch.cat(h, 1)
        else:
            h = super(SelfAttention2D, self).forward(x, key_padding_mask=padding_mask)

        """transpose h back to input shape"""
        if axis is None:
            h = h.view(R, C, B, DIM)
        elif axis == 0:
            h = h.view(R, C, B, DIM)
        else:
            h = h.view(C, R, B, DIM)
            h = h.transpose(0, 1)

        return h


class GeluFeedForward(nn.Module):
    """
    Applies a GELU-based feedforward transformation to the input tensor.

    GeluFeedForward is a neural network module that normalizes the input tensor
    and applies a two-layer feedforward network with GELU activation. This is
    commonly used in transformer architectures or other deep learning models
    to enhance the representational power of the model.
    """

    def __init__(self, input_dim: int, ff_dim: int):
        """
        A class that represents a feedforward neural network module. This module includes
        a layer normalization, a two-layer feedforward mechanism, and initialization
        of weights and biases for the second linear layer to all zeros.

        :param input_dim:
            The dimension of the input features.
        :param ff_dim:
            The dimension of the intermediate feedforward layer.
        """
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)

        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a forward pass through the sequence of operations which includes
        normalization, two linear transformations, and the application of GELU
        activation function.

        This method processes the input tensor by first normalizing it using the
        `self.norm` function. It then applies the first linear transformation
        (`self.linear1`), followed by the GELU activation, and finally the
        second linear transformation (`self.linear2`). The output is returned
        as a transformed tensor.

        :param x: Input tensor to the forward pass.
        :type x: torch.Tensor
        :return: Transformed tensor after normalization, linear transformations,
            and activation function.
        :rtype: torch.Tensor
        """
        x = self.norm(x)
        x = self.linear1(x)
        x = self.linear2(gelu(x))

        return x


@torch.jit.script
def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function on the
    input tensor. GELU is a smoother approximation to the rectified linear
    activation function and allows the model to retain more information in
    its input data. The function adds non-linearity, making it especially
    useful in deep learning architectures.

    :param x: Input tensor.
    :type x: torch.Tensor
    :return: A tensor resulting from the application of the GELU activation
        function to the input tensor. The returned tensor has the same shape
        as the input tensor.
    :rtype: torch.Tensor
    """
    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2)))
