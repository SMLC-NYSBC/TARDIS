from typing import Optional

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
from tardis.dist_pytorch.transformer.modules import gelu


class PairBiasSelfAttention(nn.Module):
    """
    SELF ATTENTION WITH EDGE FEATURE-BASED BIAS AND PRE-LAYER NORMALIZATION

    Self attention block that attend coordinate and image patches.

    Args:
        embed_dim: Number of embedded dimensions for node dimensions
        pairs_dim: Number of pairs dimensions
        num_heads: Number of heads for multi-head attention
        init_scaling: Initial scaling factor used for reset parameters
    """

    def __init__(self,
                 embed_dim,
                 pairs_dim,
                 num_heads,
                 init_scaling=1 / 1.4142135623730951):
        super().__init__()
        self.embed_dim = self.kdim = self.vdim = embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.init_scaling = init_scaling

        self.norm_input = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=False)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.pairs_dim = pairs_dim
        self.norm_pairs = nn.LayerNorm(pairs_dim)
        self.pairs_proj = nn.Linear(self.pairs_dim, self.num_heads, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)

        nn.init.xavier_uniform_(self.out_proj.weight, gain=self.init_scaling)
        # nn.init.constant_(self.out_proj.weight, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        nn.init.xavier_uniform_(self.pairs_proj.weight, gain=self.init_scaling)

    def forward(self,
                query: torch.Tensor,
                pairs: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                need_head_weights: bool = False):
        """
        Input:
            query(Nodes): Length x Batch x Channel
            pairs(Edges): Batch x Length x Length x Channel

        Args:
            query: Tensor with embedded Nodes features
            pairs: Tensor with embedded Edges features
            attn_mask: typically used to implement causal attention, where 
                the mask prevents the attention from looking forward in time
            key_padding_mask: mask to exclude keys that are pads, of shape 
                `(batch, src_len)`, where padding elements are indicated by 1s
            need_weights: return the attention weights, averaged over heads
            need_head_weights: return the attention weights for each head. 
                Return the average attention weights over all heads
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

        q = (q.contiguous().view(tgt_len,
                                 bsz * self.num_heads,
                                 self.head_dim).transpose(0, 1))
        k = (k.contiguous().view(-1,
                                 bsz * self.num_heads,
                                 self.head_dim).transpose(0, 1))
        v = (v.contiguous().view(-1,
                                 bsz * self.num_heads,
                                 self.head_dim).transpose(0, 1))
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
        assert list(attn_weights.size()) == [bsz * self.num_heads,
                                             tgt_len,
                                             src_len]

        pairs = self.norm_pairs(pairs)
        pair_weights = self.pairs_proj(pairs)

        """pair weights are B x L x L x H, needs to be B x H x L x L"""
        pair_weights = pair_weights.permute(0, 3, 1, 2)
        attn_weights = attn_weights.view(bsz,
                                         self.num_heads,
                                         tgt_len, src_len) + pair_weights
        attn_weights = attn_weights.view(bsz * self.num_heads,
                                         tgt_len,
                                         src_len)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            """don't attend to padding symbols"""
            attn_weights = attn_weights.view(bsz,
                                             self.num_heads,
                                             tgt_len,
                                             src_len)

            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn_weights = attn_weights.masked_fill(key_padding_mask,
                                                    float("-inf"))

            attn_weights = attn_weights.view(bsz * self.num_heads,
                                             tgt_len,
                                             src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = attn_weights

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads,
                                     tgt_len,
                                     self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz,
                                                   self.num_heads,
                                                   tgt_len,
                                                   src_len)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=1)

            return attn, attn_weights
        return attn


class GeluFeedForward(nn.Module):
    """
    FEED-FORWARD TRANSFORMER MODULE USING GELU

    Input: Batch x ... x Dim
    Output: Batch x ... x Dim

    Args:
        input_dim: Number of input dimension for linear transformation
        ff_dim: Number of feed forward dimension in linear transformation

    """

    def __init__(self,
                 input_dim: int,
                 ff_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, input_dim)

        nn.init.xavier_uniform_(self.linear2.weight, gain=1 / 1.4142135623730951)
        # nn.init.constant_(self.linear2.weight, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self,
                x: torch.Tensor):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.linear2(gelu(x))

        return x


class ComparisonLayer(nn.Module):
    """
    COMPARISON MODULE BETWEEN PAIRS AND NODE INPUTS

    This module convert pairs representation of dim (Length x Batch x Channels)
    it into (Batch x Length x Length x Channels) representation that can be
    compared with nodes representation.

    Args:
        input_dim: Input dimension as in pairs features
        output_dim: Output dimension as in node features
        channel_dim: Number of output channels
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 channel_dim=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, channel_dim)
        self.linear2 = nn.Linear(input_dim, channel_dim)
        self.linear3 = nn.Linear(channel_dim, output_dim)
        self.linear4 = nn.Linear(channel_dim, output_dim, bias=False)

        init_scalier = 1 / 1.4142135623730951

        nn.init.xavier_uniform_(self.linear3.weight, gain=init_scalier)
        # nn.init.constant_(self.linear3.weight, 0.0)
        nn.init.constant_(self.linear3.bias, 0.0)

        nn.init.xavier_uniform_(self.linear4.weight, gain=init_scalier)
        # nn.init.constant_(self.linear4.weight, 0.0)

    def forward(self,
                x: torch.Tensor):
        """Length x Batch x In_Channel"""
        x = x.transpose(0, 1)
        x = self.norm(x)
        a = self.linear1(x)
        b = self.linear2(x)

        """Batch x Length x Length x Out_Channels"""
        return self.linear3(a.unsqueeze(2) * b.unsqueeze(1)) + \
            self.linear4(a.unsqueeze(2) - b.unsqueeze(1))


class QuadraticEdgeUpdate(nn.Module):
    """
    QUADRATIC UPDATE MODEL FOR NODES FEATURES

    This module take node feature representation and perform quadratic
    attention for each points.

    Args:
        input_dim: Number of input channels
        channel_dim: Number of output channels
        axis: Indicate axis around which the attention is given
    """

    def __init__(self,
                 input_dim,
                 channel_dim=128,
                 axis=1):
        super().__init__()
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.init_scaling = 1 / 1.4142135623730951

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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_a.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_a.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_b.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_b.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_c.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_c.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_d.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_d.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_o.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_o.bias, 0.0)

    def forward(self,
                z: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        z = self.norm_input(z)

        a = torch.sigmoid(self.gate_a(z)) * self.linear_a(z)  # B x L x L x O
        b = torch.sigmoid(self.gate_b(z)) * self.linear_b(z)  # B x L x L x O
        c = torch.sigmoid(self.gate_c(z)) * self.linear_c(z)  # B x L x L x O
        d = torch.sigmoid(self.gate_d(z)) * self.linear_d(z)  # B x L x L x O

        if mask is not None:
            mask = mask.unsqueeze(3).expand(mask.size(0),
                                            mask.size(1),
                                            mask.size(2),
                                            a.size(3))
            a = torch.where(mask, torch.zeros_like(a), a)
            b = torch.where(mask, torch.zeros_like(b), b)
            c = torch.where(mask, torch.zeros_like(c), c)
            d = torch.where(mask, torch.zeros_like(d), d)

        # i,j -> i,k j,k, il, jl
        if self.axis == 1:
            k = torch.einsum('biko,bjko,bilo,bjlo->bijo', a, b, c, d)
        elif self.axis == 0:
            k = torch.einsum('bkio,bkjo,blio,bljo->bijo', a, b, c, d)

        return torch.sigmoid(self.gate_o(z)) * self.linear_o(self.norm_o(k))


class TriangularEdgeUpdate(nn.Module):
    """
    TRIANGULAR UPDATE MODEL FOR NODES FEATURES

    This module take node feature representation and perform triangular
    attention for each points. Similar as in Alphafold 2 approach.

    Args:
        input_dim: Number of input channels
        channel_dim: Number of output channels
        axis: Indicate axis around which the attention is given
    """

    def __init__(self,
                 input_dim,
                 channel_dim=128,
                 axis=1):
        super().__init__()
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.axis = axis
        self.init_scaling = 1 / 1.4142135623730951

        self.norm_input = nn.LayerNorm(input_dim)

        self.linear_a = nn.Linear(input_dim, channel_dim)
        self.gate_a = nn.Linear(input_dim, channel_dim)

        self.linear_b = nn.Linear(input_dim, channel_dim)
        self.gate_b = nn.Linear(input_dim, channel_dim)

        self.norm_o = nn.LayerNorm(channel_dim)
        self.gate_o = nn.Linear(input_dim, input_dim)
        self.linear_o = nn.Linear(channel_dim, input_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_a.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_a.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_b.weight, gain=self.init_scaling)
        nn.init.constant_(self.linear_b.bias, 0.0)

        nn.init.xavier_uniform_(self.linear_o.weight, gain=self.init_scaling)
        # nn.init.constant_(self.linear_o.weight, 0.0)
        nn.init.constant_(self.linear_o.bias, 0.0)

    def forward(self,
                z: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        z = self.norm_input(z)

        a = torch.sigmoid(self.gate_a(z)) * self.linear_a(z)  # B x L x L x O
        b = torch.sigmoid(self.gate_b(z)) * self.linear_b(z)  # B x L x L x O

        if mask is not None:
            mask = mask.unsqueeze(3).expand(mask.size(0),
                                            mask.size(1),
                                            mask.size(2),
                                            a.size(3))
            a = torch.where(mask, torch.zeros_like(a), a)
            b = torch.where(mask, torch.zeros_like(b), b)

        # i,j -> i,k j,k
        if self.axis == 1:
            k = torch.einsum('biko,bjko->bijo', a, b)
        elif self.axis == 0:
            k = torch.einsum('bkio,bkjo->bijo', a, b)

        return torch.sigmoid(self.gate_o(z)) * self.linear_o(self.norm_o(k))


class MultiHeadAttention(nn.Module):
    """
    MULTI-HEADED ATTENTION

    See "Attention Is All You Need" for more details. Modified from fairseq.

    Args:
        embed_dim: Number of embedded dimensions for node features
        num_heads: Number of heads for multi-head attention
        kdim: Key dimensions.
        vdim: Values dimensions.
        dropout: Dropout probability.
        bias: If True add bias.
        add_bias_kv: If True add bias for key and values.
        add_zero_attn: If True replace attention with zero-out mask.
        self_attention: If True self attention is used.
        encoder_decoder_attention: If True self attention over encode/decoder
            is used.
        init_scaling: Initial scaling factor used for reset parameters.
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 kdim=None,
                 vdim=None,
                 dropout=0,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 self_attention=False,
                 encoder_decoder_attention=False,
                 init_scaling=1 / 1.4142135623730951):

        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert (self.head_dim * num_heads == self.embed_dim), \
            "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, \
            "Self-attention requires query, key and value to be of the same size"

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

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self):
        self.tpu = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=self.init_scaling)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=self.init_scaling)

        nn.init.xavier_uniform_(self.out_proj.weight, gain=self.init_scaling)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k, gain=self.init_scaling)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v, gain=self.init_scaling)

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False,
                static_kv: bool = False,
                attn_mask: Optional[torch.Tensor] = None,
                before_softmax: bool = False,
                need_head_weights: bool = False):
        """
        Input: Batch x Length x Length x Dim

        Args:
            query: Query input.
            key: Key input.
            value: Value input.
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            static_kv: If True previous key_padding_mask is used for attention
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
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
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask,
                     key_padding_mask.new_zeros(key_padding_mask.size(0), 1)],
                    dim=1,
                )

        q = (q.contiguous().view(tgt_len,
                                 bsz * self.num_heads,
                                 self.head_dim).transpose(0, 1))
        if k is not None:
            k = (k.contiguous().view(-1,
                                     bsz * self.num_heads,
                                     self.head_dim).transpose(0, 1))
        if v is not None:
            v = (v.contiguous().view(-1,
                                     bsz * self.num_heads,
                                     self.head_dim).transpose(0, 1))

        assert k is not None
        src_len = k.size(1)

        """This is part of a workaround to get around fork/join parallelism
        not supporting Optional types."""
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])],
                          dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])],
                          dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask,
                     torch.zeros(key_padding_mask.size(0),
                                 1).type_as(key_padding_mask)],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights,
                                              tgt_len,
                                              src_len,
                                              bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads,
                                             tgt_len,
                                             src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            """don't attend to padding symbols"""
            attn_weights = attn_weights.view(bsz,
                                             self.num_heads,
                                             tgt_len,
                                             src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask,
                                                        float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads,
                                             tgt_len,
                                             src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = F.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads,
                                     tgt_len,
                                     self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            """
            When ONNX tracing a single decoder step (sequence length == 1)
            the transpose is a no-op copy before view, thus unnecessary
            """
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len,
                                                          bsz,
                                                          embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[torch.Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            if not need_head_weights:
                """average attention weights over heads"""
                attn_weights = attn_weights.mean(dim=1)

            return attn, attn_weights

        return attn

    # @staticmethod
    # def _append_prev_key_padding_mask(key_padding_mask: Optional[torch.Tensor],
    #                                   prev_key_padding_mask: Optional[torch.Tensor],
    #                                   batch_size: int,
    #                                   src_len: int,
    #                                   static_kv: bool) -> Optional[torch.Tensor]:
    #     """saved key padding masks have shape (bsz, seq_len)"""
    #     if prev_key_padding_mask is not None and static_kv:
    #         new_key_padding_mask = prev_key_padding_mask
    #     elif prev_key_padding_mask is not None and key_padding_mask is not None:
    #         new_key_padding_mask = torch.cat(
    #             [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
    #         )
    #         """
    #         During incremental decoding, as the padding token enters and
    #         leaves the frame, there will be a time when prev or current
    #         is None
    #         """
    #     elif prev_key_padding_mask is not None:
    #         filler = torch.zeros(
    #             (batch_size, src_len - prev_key_padding_mask.size(1)),
    #             device=prev_key_padding_mask.device,
    #         )
    #         new_key_padding_mask = torch.cat(
    #             [prev_key_padding_mask.float(), filler.float()], dim=1
    #         )
    #     elif key_padding_mask is not None:
    #         filler = torch.zeros(
    #             (batch_size, src_len - key_padding_mask.size(1)),
    #             device=key_padding_mask.device,
    #         )
    #         new_key_padding_mask = torch.cat(
    #             [filler.float(), key_padding_mask.float()], dim=1
    #         )
    #     else:
    #         new_key_padding_mask = prev_key_padding_mask
    #     return new_key_padding_mask

    # def apply_sparse_mask(self,
    #                       attn_weights,
    #                       tgt_len,
    #                       src_len,
    #                       bsz):
    #     return attn_weights

    # def upgrade_state_dict_named(self, state_dict, name):
    #     prefix = name + "." if name != "" else ""
    #     items_to_add = {}
    #     keys_to_remove = []
    #     for k in state_dict.keys():
    #         if k.endswith(prefix + "in_proj_weight"):
    #             """in_proj_weight used to be q + k + v with same dimensions"""
    #             dim = int(state_dict[k].shape[0] / 3)
    #             items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
    #             items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
    #             items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

    #             keys_to_remove.append(k)

    #             k_bias = prefix + "in_proj_bias"
    #             if k_bias in state_dict.keys():
    #                 dim = int(state_dict[k].shape[0] / 3)
    #                 items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
    #                 items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim: 2 * dim]
    #                 items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

    #                 keys_to_remove.append(prefix + "in_proj_bias")

    #     for k in keys_to_remove:
    #         del state_dict[k]

    #     for key, value in items_to_add.items():
    #         state_dict[key] = value


class SelfAttention2D(MultiHeadAttention):
    """
    COMPUTE SELF-ATTENTION OVER 2D INPUT

    Perform self-attention over 2D input for node features using multi-head
    attention.

    Args:
        embed_dim: Number of embedded dimensions for node features
        num_heads: Number of heads for multi-head attention
        axis: Indicate axis over which the attention is perform
        dropout: Dropout probability
        max_size: Maximum size of batch
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 axis=None,
                 dropout=0,
                 max_size=4194304):
        super(SelfAttention2D, self).__init__(embed_dim,
                                              num_heads,
                                              dropout=dropout,
                                              self_attention=True)
        self.axis = axis
        self.max_size = max_size

    def forward(self,
                x: torch.Tensor,
                padding_mask=None):
        """
        x : num_rows X num_cols X batch_size X embed_dim
        padding_mask : batch_size X num_rows X num_cols
        """

        N, M, B, H = x.size()

        """
        reshape X depending on axis attention mode!
        flatten over rows and cols for full N*M*N*M attention
        """
        axis = self.axis
        if axis is None:
            x = x.view(N * M, B, H)
            if padding_mask is not None:
                padding_mask = padding_mask.view(B, N * M)
        else:
            assert axis == 0 or axis == 1

            """attend along the row dimension"""
            if axis == 0:
                x = x.view(N, M * B, H)
                if padding_mask is not None:
                    padding_mask = padding_mask.permute(2, 0, 1)
                    padding_mask = padding_mask.reshape(M * B, N)
                """attend along the col dimension"""
            else:
                x = x.transpose(0, 1)
                x = x.reshape(M, N * B, H)
                if padding_mask is not None:
                    padding_mask = padding_mask.permute(1, 0, 2)
                    padding_mask = padding_mask.reshape(N * B, M)

        if 0 < self.max_size < x.size(0) ** 2 * x.size(1):
            """
            Attention matrix size times batch size will exceed maximum
            allowable entries split into batches to make attention matrix RAM
            workable calculating attention over batches helps reduce RAM when
            N or M are large
            """
            batch_size = x.size(0) ** 2 // self.max_size
            if batch_size < 1:
                """might run out of RAM, but batch size can't be < 1"""
                batch_size = 1

            h = []
            for i in range(0, x.size(1), batch_size):
                xi = x[:, i:i + batch_size]
                mask = None
                if padding_mask is not None:
                    mask = padding_mask[i:i + batch_size]
                h.append(super(SelfAttention2D,
                               self).forward(xi,
                                             key_padding_mask=mask))
            h = torch.cat(h, 1)
        else:
            h = super(SelfAttention2D,
                      self).forward(x,
                                    key_padding_mask=padding_mask)

        """transpose h back to input shape"""
        if axis is None:
            h = h.view(N, M, B, H)
        elif axis == 0:
            h = h.view(N, M, B, H)
        else:
            h = h.view(M, N, B, H)
            h = h.transpose(0, 1)

        return h
