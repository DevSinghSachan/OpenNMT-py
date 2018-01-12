"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import onmt
from onmt.Models import EncoderBase
from onmt.Models import DecoderState
from onmt.Utils import aeq

MAX_SIZE = 5000


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """
    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = onmt.modules.BottleLinear(size, hidden_size)
        self.w_2 = onmt.modules.BottleLinear(hidden_size, size)
        self.layer_norm = onmt.modules.BottleLayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """

    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = onmt.modules.MultiHeadedAttention(
            head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm = onmt.modules.BottleLayerNorm(size)

    def forward(self, input, mask):
        input_norm = self.layer_norm(input)
        mid, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.feed_forward(mid + input)
        return out


def seq_func(func, x, reconstruct_shape=True):
    """Change implicitly function's input x from ndim=3 to ndim=2

    :param func: function to be applied to input x
    :param x: Tensor of batched sentence level word features
    :param reconstruct_shape: boolean, if the output needs to be of the same shape as input x
    :return: Tensor of shape (batchsize, dimension, sentence_length) or (batchsize x sentence_length, dimension)
    """
    batch, units, length = x.shape
    e = torch.transpose(x, 1, 2).contiguous().view(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = torch.transpose(e.view((batch, length, out_units)), 1, 2).contiguous()
    assert (e.shape == (batch, out_units, length))
    return e


class LayerNorm(nn.Module):
    """Layer normalization module.
    Code adapted from OpenNMT-py open-source toolkit on 08/01/2018:
    URL: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/UtilClass.py#L24"""
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) + self.b_2.expand_as(ln_out)
        return ln_out


class LayerNormSent(LayerNorm):
    """Position-wise layer-normalization layer for array of shape (batchsize, dimension, sentence_length)."""
    def __init__(self, n_units, eps=1e-3):
        super(LayerNormSent, self).__init__(n_units, eps=eps)

    def forward(self, x):
        y = seq_func(super(LayerNormSent, self).forward, x)
        return y


class LinearSent(nn.Module):
    """Position-wise Linear Layer for sentence block. array of shape (batchsize, dimension, sentence_length)."""
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSent, self).__init__()
        self.L = nn.Linear(input_dim, output_dim, bias=bias)
        self.L.weight.data.uniform_(-3./input_dim, 3./input_dim)
        if bias:
            self.L.bias.data.fill_(0.)
        self.output_dim = output_dim

    def forward(self, x):
        output = seq_func(self.L, x)
        return output


class MultiHeadAttention(nn.Module):
    """Multi Head Attention Layer for Sentence Blocks. For computational efficiency, dot-product to calculate
    query-key scores is performed in all the heads together."""
    def __init__(self, n_units, multi_heads=8, attn_dropout=False, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = LinearSent(n_units, n_units, bias=False)
        self.W_K = LinearSent(n_units, n_units, bias=False)
        self.W_V = LinearSent(n_units, n_units, bias=False)
        self.finishing_linear_layer = LinearSent(n_units, n_units, bias=False)
        self.h = multi_heads
        self.scale_score = 1. / (n_units // multi_heads) ** 0.5
        self.attn_dropout = attn_dropout
        if attn_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, z=None, mask=None):
        h = self.h
        Q = self.W_Q(x)
        if z is None:
            K, V = self.W_K(x), self.W_V(x)
        else:
            K, V = self.W_K(z), self.W_V(z)

        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate attention scores with mask for zero-padded areas
        # Perform multi-head attention using pseudo batching all together at once for efficiency
        Q = torch.cat(torch.chunk(Q, h, dim=1), dim=0)
        K = torch.cat(torch.chunk(K, h, dim=1), dim=0)
        V = torch.cat(torch.chunk(V, h, dim=1), dim=0)

        assert (Q.shape == (batch * h, n_units // h, n_querys))
        assert (K.shape == (batch * h, n_units // h, n_keys))
        assert (V.shape == (batch * h, n_units // h, n_keys))

        mask = torch.cat([mask] * h, dim=0)
        Q = Q.transpose(1, 2).contiguous()
        batch_A = torch.bmm(Q, K) * self.scale_score

        batch_A = batch_A.masked_fill(1. - mask, -np.inf)
        batch_A = F.softmax(batch_A, dim=2)

        # Replaces 'NaN' with zeros and other values with the original ones
        batch_A = batch_A.masked_fill(batch_A != batch_A, 0.)
        assert (batch_A.shape == (batch * h, n_querys, n_keys))

        # probabiliy, B x T x S
        top_attns = torch.cat(torch.chunk(batch_A, h, dim=0), dim=1)

        # Attention Dropout
        if self.attn_dropout:
            batch_A = self.dropout(batch_A)

        # Calculate Weighted Sum
        V = V.transpose(1, 2).contiguous()
        C = torch.transpose(torch.bmm(batch_A, V), 1, 2).contiguous()
        assert (C.shape == (batch * h, n_units // h, n_querys))

        # Joining the Multiple Heads
        C = torch.cat(torch.chunk(C, h, dim=0), dim=1)
        assert (C.shape == (batch, n_units, n_querys))

        # Final linear layer
        C = self.finishing_linear_layer(C)
        return C, top_attns


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        self.W_1 = LinearSent(n_units, n_inner_units)
        self.act = nn.ReLU()
        self.W_2 = LinearSent(n_inner_units, n_units)

    def forward(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class TransformerEncoderLayerNew(nn.Module):
    def __init__(self, n_units, multi_heads=8, dropout=0.2, layer_norm=True, attn_dropout=False):
        super(TransformerEncoderLayerNew, self).__init__()
        self.layer_norm = layer_norm
        self.self_attention = MultiHeadAttention(n_units, multi_heads, attn_dropout, dropout)
        self.dropout1 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_1 = LayerNormSent(n_units, eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units)
        self.dropout2 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_2 = LayerNormSent(n_units, eps=1e-3)

    def forward(self, e, xx_mask):
        sub, _ = self.self_attention(e, mask=xx_mask)
        e = e + self.dropout1(sub)
        if self.layer_norm:
            e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + self.dropout2(sub)
        if self.layer_norm:
            e = self.ln_2(e)
        return e


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O



    Args:
       num_layers (int): number of encoder layers
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    """
    # def __init__(self, num_layers, hidden_size,
    #              dropout, embeddings):
    #     super(TransformerEncoder, self).__init__()
    #
    #     self.num_layers = num_layers
    #     self.embeddings = embeddings
    #     self.transformer = nn.ModuleList(
    #         [TransformerEncoderLayer(hidden_size, dropout)
    #          for i in range(num_layers)])
    #     self.layer_norm = onmt.modules.BottleLayerNorm(hidden_size)
    #
    # def forward(self, input, lengths=None, hidden=None):
    #     """ See :obj:`EncoderBase.forward()`"""
    #     self._check_args(input, lengths, hidden)
    #
    #     emb = self.embeddings(input)
    #     s_len, n_batch, emb_dim = emb.size()
    #
    #     out = emb.transpose(0, 1).contiguous()
    #     words = input[:, :, 0].transpose(0, 1)
    #     # CHECKS
    #     out_batch, out_len, _ = out.size()
    #     w_batch, w_len = words.size()
    #     aeq(out_batch, w_batch)
    #     aeq(out_len, w_len)
    #     # END CHECKS
    #
    #     # Make mask.
    #     padding_idx = self.embeddings.word_padding_idx
    #     mask = words.data.eq(padding_idx).unsqueeze(1) \
    #         .expand(w_batch, w_len, w_len)
    #
    #     # Run the forward pass of every layer of the tranformer.
    #     for i in range(self.num_layers):
    #         out = self.transformer[i](out, mask)
    #     out = self.layer_norm(out)
    #
    #     return Variable(emb.data), out.transpose(0, 1).contiguous()

    def __init__(self, num_layers, hidden_size, dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.layers = torch.nn.ModuleList()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList([TransformerEncoderLayerNew(n_units=hidden_size,
                                                                     dropout=dropout)
                                          for i in range(num_layers)])

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()

        out = emb.transpose(0, 1).contiguous()
        out = out.transpose(1, 2).contiguous()

        words = input[:, :, 0].transpose(0, 1)
        # CHECKS
        out_batch, _, out_len = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        # END CHECKS

        # Make mask.
        padding_idx = self.embeddings.word_padding_idx
        mask = make_attention_mask(words, words, padding_idx)

        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)

        out = out.transpose(1, 2).contiguous()
        return Variable(emb.data), out.transpose(0, 1).contiguous()


def make_attention_mask(source_block, target_block, padding_idx):
    mask = (target_block[:, None, :] > padding_idx) * (source_block[:, :, None] > padding_idx)
    # (batch, source_length, target_length)
    return mask


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      droput(float): dropout probability(0-1.0).
      head_count(int): the number of heads for MultiHeadedAttention.
      hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """
    def __init__(self, size, dropout,
                 head_count=8, hidden_size=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(
                head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size,
                                                    hidden_size,
                                                    dropout)
        self.layer_norm_1 = onmt.modules.BottleLayerNorm(size)
        self.layer_norm_2 = onmt.modules.BottleLayerNorm(size)
        self.dropout = dropout
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)

    def forward(self, input, context, src_pad_mask, tgt_pad_mask):
        # Args Checks
        input_batch, input_len, _ = input.size()
        contxt_batch, contxt_len, _ = context.size()
        aeq(input_batch, contxt_batch)

        src_batch, t_len, s_len = src_pad_mask.size()
        tgt_batch, t_len_, t_len__ = tgt_pad_mask.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        aeq(t_len, t_len_, t_len__, input_len)
        aeq(s_len, contxt_len)
        # END Args Checks

        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1),
                            :tgt_pad_mask.size(1)]
                            .expand_as(tgt_pad_mask), 0)
        input_norm = self.layer_norm_1(input)
        query, attn = self.self_attn(input_norm, input_norm, input_norm,
                                     mask=dec_mask)
        query_norm = self.layer_norm_2(query+input)
        mid, attn = self.context_attn(context, context, query_norm,
                                      mask=src_pad_mask)
        output = self.feed_forward(mid+query+input)

        # CHECKS
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)

        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        # END CHECKS

        return output, attn

    def _get_attn_subsequent_mask(self, size):
        ''' Get an attention mask to avoid using the subsequent info.'''
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoderLayerNew(nn.Module):
    def __init__(self, n_units, multi_heads=8, dropout=0.2, layer_norm=True, attn_dropout=False):
        super(TransformerDecoderLayerNew, self).__init__()
        self.layer_norm = layer_norm
        self.self_attention = MultiHeadAttention(n_units, multi_heads, attn_dropout, dropout)
        self.dropout1 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_1 = LayerNormSent(n_units, eps=1e-3)

        self.source_attention = MultiHeadAttention(n_units, multi_heads, attn_dropout, dropout)
        self.dropout2 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_2 = LayerNormSent(n_units, eps=1e-3)

        self.feed_forward = FeedForwardLayer(n_units)
        self.dropout3 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_3 = LayerNormSent(n_units, eps=1e-3)

    def forward(self, e, s, xy_mask, yy_mask):
        sub, _ = self.self_attention(e, mask=yy_mask)
        e = e + self.dropout1(sub)
        if self.layer_norm:
            e = self.ln_1(e)

        sub, top_attns = self.source_attention(e, s, mask=xy_mask)
        e = e + self.dropout2(sub)
        if self.layer_norm:
            e = self.ln_2(e)

        sub = self.feed_forward(e)
        e = e + self.dropout3(sub)
        if self.layer_norm:
            e = self.ln_3(e)
        return e, top_attns


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

       attn_type (str): if using a seperate copy attention
    """
    # def __init__(self, num_layers, hidden_size, attn_type,
    #              copy_attn, dropout, embeddings):
    #     super(TransformerDecoder, self).__init__()
    #
    #     # Basic attributes.
    #     self.decoder_type = 'transformer'
    #     self.num_layers = num_layers
    #     self.embeddings = embeddings
    #
    #     # Build TransformerDecoder.
    #     self.transformer_layers = nn.ModuleList([TransformerDecoderLayer(hidden_size, dropout) for _ in range(num_layers)])
    #
    #     # TransformerDecoder has its own attention mechanism.
    #     # Set up a separated copy attention layer, if needed.
    #     self._copy = False
    #     if copy_attn:
    #         self.copy_attn = onmt.modules.GlobalAttention(
    #             hidden_size, attn_type=attn_type)
    #         self._copy = True
    #     self.layer_norm = onmt.modules.BottleLayerNorm(hidden_size)

    # def forward(self, input, context, state, context_lengths=None):
    #     """
    #     See :obj:`onmt.modules.RNNDecoderBase.forward()`
    #     """
    #     # CHECKS
    #     assert isinstance(state, TransformerDecoderState)
    #     input_len, input_batch, _ = input.size()
    #     contxt_len, contxt_batch, _ = context.size()
    #     aeq(input_batch, contxt_batch)
    #
    #     if state.previous_input is not None:
    #         input = torch.cat([state.previous_input, input], 0)
    #
    #     src = state.src
    #     src_words = src[:, :, 0].transpose(0, 1)
    #     tgt_words = input[:, :, 0].transpose(0, 1)
    #     src_batch, src_len = src_words.size()
    #     tgt_batch, tgt_len = tgt_words.size()
    #     aeq(input_batch, contxt_batch, src_batch, tgt_batch)
    #     aeq(contxt_len, src_len)
    #     # aeq(input_len, tgt_len)
    #     # END CHECKS
    #
    #     # Initialize return variables.
    #     outputs = []
    #     attns = {"std": []}
    #     if self._copy:
    #         attns["copy"] = []
    #
    #     # Run the forward pass of the TransformerDecoder.
    #     emb = self.embeddings(input)
    #     assert emb.dim() == 3  # len x batch x embedding_dim
    #
    #     output = emb.transpose(0, 1).contiguous()
    #     src_context = context.transpose(0, 1).contiguous()
    #
    #     padding_idx = self.embeddings.word_padding_idx
    #     src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
    #         .expand(src_batch, tgt_len, src_len)
    #     tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
    #         .expand(tgt_batch, tgt_len, tgt_len)
    #
    #     for i in range(self.num_layers):
    #         output, attn \
    #             = self.transformer_layers[i](output, src_context,
    #                                          src_pad_mask, tgt_pad_mask)
    #
    #     output = self.layer_norm(output)
    #     # Process the result and update the attentions.
    #     outputs = output.transpose(0, 1).contiguous()
    #     if state.previous_input is not None:
    #         outputs = outputs[state.previous_input.size(0):]
    #         attn = attn[:, state.previous_input.size(0):].squeeze()
    #         attn = torch.stack([attn])
    #     attns["std"] = attn
    #     if self._copy:
    #         attns["copy"] = attn
    #
    #     # Update the state.
    #     state.update_state(input)
    #
    #     return outputs, state, attns

    def init_decoder_state(self, src, context, enc_hidden):
        return TransformerDecoderState(src)

    def __init__(self, num_layers, hidden_size, attn_type, copy_attn, dropout, embeddings):
        super(TransformerDecoder, self).__init__()
        # Basic attributes.
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings

        # Build TransformerDecoder.
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayerNew(n_units=hidden_size, dropout=dropout) for _ in range(num_layers)])

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(
                hidden_size, attn_type=attn_type)
            self._copy = True

    def forward(self, input, context, state, context_lengths = None):
        # CHECKS
        assert isinstance(state, TransformerDecoderState)
        input_len, input_batch, _ = input.size()
        contxt_len, contxt_batch, _ = context.size()
        aeq(input_batch, contxt_batch)

        if state.previous_input is not None:
            input = torch.cat([state.previous_input, input], 0)

        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = input[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        aeq(contxt_len, src_len)
        # aeq(input_len, tgt_len)
        # END CHECKS

        # Initialize return variables.
        outputs = []
        attns = {"std": []}
        if self._copy:
            attns["copy"] = []

        # Run the forward pass of the TransformerDecoder.
        emb = self.embeddings(input)
        assert emb.dim() == 3  # len x batch x embedding_dim

        output = emb.transpose(0, 1).contiguous()
        output = output.transpose(1, 2).contiguous()

        src_context = context.transpose(0, 1).contiguous()
        src_context = src_context.transpose(1, 2).contiguous()

        padding_idx = self.embeddings.word_padding_idx
        xy_mask = make_attention_mask(tgt_words, src_words, padding_idx)
        yy_mask = make_attention_mask(tgt_words, tgt_words, padding_idx)
        yy_mask *= make_history_mask(tgt_words)

        for i in range(self.num_layers):
            output, attn = self.transformer_layers[i](output, src_context, xy_mask, yy_mask)

        # attn.shape = torch.Size([100, 27, 9]) = [B x trg x src]

        # Process the result and update the attentions.
        outputs = output.transpose(1, 2).contiguous()
        outputs = outputs.transpose(0, 1).contiguous()

        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])
        attns["std"] = attn
        if self._copy:
            attns["copy"] = attn

        # Update the state.
        state.update_state(input)
        return outputs, state, attns

if torch.cuda.is_available():
    BYTE_TYPE = torch.cuda.ByteTensor
else:
    BYTE_TYPE = torch.ByteTensor


def make_history_mask(block):
    batch, length = block.shape
    arange = np.arange(length)
    history_mask = (arange[None,] <= arange[:, None])[None,]
    history_mask = np.broadcast_to(history_mask, (batch, length, length))
    history_mask = history_mask.astype(np.int32)
    history_mask = Variable(torch.ByteTensor(history_mask).type(BYTE_TYPE))
    return history_mask


class TransformerDecoderState(DecoderState):
    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return (self.previous_input, self.src)

    def update_state(self, input):
        """ Called for every decoder forward pass. """
        self.previous_input = input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1),
                            volatile=True)
