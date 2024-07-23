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
    SELF-ATTENTION WITH EDGE FEATURE-BASED BIAS AND PRE-LAYER NORMALIZATION

    Self-attention block that attends coordinate and image patches or RGB.
    """

    def __init__(
        self,
        embed_dim: int,
        pairs_dim: int,
        num_heads: int,
        init_scaling=1 / 1.4142135623730951,
    ):
        """

        Args:
            embed_dim (int): Number of embedded dimensions for node dimensions.
            pairs_dim (int): Number of pairs dimensions.
            num_heads (int): Number of heads for multi-head attention.
            init_scaling (float): Initial scaling factor used for reset parameters.
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
        Initial parameter and bias scaling.
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
        Forward attention over node features.

        Args:
            query (torch.Tensor): Nodes features  [Length x Batch x Channel].
            pairs (torch.Tensor): Edges features [Batch x Length x Length x Channel].
            attn_mask (torch.Tensor): Typically used to implement causal attention,
                where the mask prevents the attention from looking forward in time.
            key_padding_mask (torch.Tensor): Mask to exclude keys that are pads,
                of shape [Batch, src_len]
            need_weights (bool): If True, return the attention weights,
                and averaged overheads.
            need_head_weights (bool): If True, return the attention weights
                for each head.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: Attention tensor
            for node features.
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
    COMPARISON MODULE BETWEEN PAIRS AND NODE INPUTS

    This module converts pairs representation of dim (Length x Batch x Channels)
    into (Batch x Length x Length x Channels) representation that can be compared
    with node representation.

    Args:
        input_dim (int): Input dimension as in pairs features.
        output_dim (int): Output dimension as in node features.
        channel_dim (int): Number of output channels.
    """

    def __init__(self, input_dim: int, output_dim: int, channel_dim=128):
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
        Forward node compatible layer.

        Args:
            x (torch.Tensor): Node features after attention layer.

        Returns:
            torch.Tensor: Converted Node features to [Batch x Length x Length x
            Out_Channels] shape.
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
    TRIANGULAR UPDATE MODEL FOR NODES FEATURES

    This module takes node feature representation and performs triangular
    attention for each point. Similar to in Alphafold2 approach.

    Args:
        input_dim (int): Number of input channels.
        channel_dim (int): Number of output channels.
        axis (int): Indicate the axis around which the attention is given.
    """

    def __init__(self, input_dim, channel_dim=128, axis=1):
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
        Initial parameter and bias scaling.
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
        Forward Triangular edge update.

        Args:
            z (torch.Tensor): Edge features.
            mask (torch.Tensor, optional): Optional mask torch.Tensor layer.

        Returns:
            torch.Tensor: Updated edge features.
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
    QUADRATIC UPDATE MODEL FOR NODES FEATURES

    This module takes node feature representation and performs quadratic
    attention for each point. This is a modified Alphafold2 solution.

    Args:
        input_dim (int): Number of input channels.
        channel_dim (int): Number of output channels.
        axis (int): Indicate the axis around which the attention is given.
    """

    def __init__(self, input_dim, channel_dim=128, axis=1):
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
        Initial parameter and bias scaling.
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
        Forward Quadratic edge update.

        Args:
            z (torch.Tensor): Edge features.
            mask (torch.Tensor, optional): Optional mask torch.Tensor layer.

        Returns:
            torch.Tensor: Updated edge features.
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
    MULTI-HEADED ATTENTION

    See "Attention Is All You Need" for more details. Modified from 'fairseq'.

    Args:
        embed_dim (int): Number of embedded dimensions for node features.
        num_heads (int): Number of heads for multi-head attention.
        kdim: Key dimensions.
        vdim: Values dimensions.
        dropout (float): Dropout probability.
        bias (bool): If True add bias.
        add_bias_kv (bool): If True add bias for keys and values.
        add_zero_attn (bool): If True replace attention with a zero-out mask.
        self_attention (bool): If True self-attention is used.
        encoder_decoder_attention (bool): If True self-attention over
            encode/decoder is used.
        init_scaling (float): The initial scaling factor used for reset parameters.
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
        Initial parameter and bias scaling.
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
        Forward for MHA.

        Args:
            query (torch.Tensor): Query input.
            key (torch.Tensor, optional): Key input.
            value (torch.Tensor, optional): Value input.
            key_padding_mask (torch.Tensor, optional): Mask to exclude keys that
                are pads, of shape `(batch, src_len)`.
            need_weights (bool, optional): Return the attention weights,
                averaged overheads.
            attn_mask (torch.Tensor, optional): Typically used to implement
                causal attention.
            before_softmax (bool, optional): Return the raw attention weights and
                values before the attention softmax.
            need_head_weights (bool, optional): Return the attention weights for
                each head. Implies *need_weights*.
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
    COMPUTE SELF-ATTENTION OVER 2D INPUT

    Perform self-attention over 2D input for node features using multi-head
    attention.

    Args:
        embed_dim (int): Number of embedded dimensions for node features.
        num_heads (int): Number of heads for multi-head attention.
        axis (int): Indicate the axis over which the attention is performed.
        dropout (float): Dropout probability.
        max_size (int): Maximum size of the batch.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, axis=None, dropout=0.0, max_size=4194304
    ):
        super(SelfAttention2D, self).__init__(
            embed_dim, num_heads, dropout=dropout, self_attention=True
        )
        self.axis = axis
        self.max_size = max_size

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        """
        Forward self-attention over 2D-edge features.

        Reshape X depending on the axis attention mode!
        flatten over rows and cols for full N*M*N*M attention.

        Args:
            x (torch.Tensor): Edge feature self-attention update.
                [num_rows X num_cols X batch_size X embed_dim].
            padding_mask (torch.Tensor): Optional padding mask.
                [batch_size X num_rows X num_cols].
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
    FEED-FORWARD TRANSFORMER MODULE USING GELU

    Input: Batch x ... x Dim
    Output: Batch x ... x Dim

    Args:
        input_dim (int): Number of input dimensions for linear transformation.
        ff_dim (int): Number of feed-forward dimensions in linear transformation.
    """

    def __init__(self, input_dim: int, ff_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)

        nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward Gelu normalized tensor.

        Args:
            x (torch.Tensor): Any Tensor of shape [B x ... x D].

        Returns:
            torch.Tensor: Gelu normalized tensor.
        """
        x = self.norm(x)
        x = self.linear1(x)
        x = self.linear2(gelu(x))

        return x


@torch.jit.script
def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    CUSTOM GAUSSIAN ERROR LINEAR UNITS ACTIVATION FUNCTION

    Args:
        x (torch.Tensor): Tensor input for activation.

    Returns:
        torch.Tensor: Gelu transform Tensor.
    """
    return x * 0.5 * (1.0 + torch.erf(x / sqrt(2)))
