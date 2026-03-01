# encoding: utf-8
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math
import scipy.stats as stats

import utils
import decoding
import preprocess
from pad_utils import PadRemover

cudnn.benchmark = True


def input_like(tensor, val=0):
    """
    Use clone() + fill_() to make sure that a tensor ends up on the right
    device at runtime.
    """
    return tensor.clone().fill_(val)


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):
    """Outputs random values from a truncated normal distribution.
    The generated values follow a normal distribution with specified mean
    and standard deviation, except that values whose magnitude is more
    than 2 standard deviations from the mean are dropped and re-picked.
    API from: https://www.tensorflow.org/api_docs/python/tf/truncated_normal
    """
    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev,
                        (upper - mean) / stddev,
                        loc=mean,
                        scale=stddev)
    values = X.rvs(size=shape)
    return torch.from_numpy(values.astype(dtype))


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a truncated normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters using Truncated Normal Initializer (default in Tensorflow)
        """
        # Initialize the embedding parameters (Default)
        # This works well too
        # self.embed_word.weight.data.uniform_(-3. / self.num_embeddings,
        #                                      3. / self.num_embeddings)

        self.weight.data = truncated_normal(shape=(self.num_embeddings,
                                                   self.embedding_dim),
                                            stddev=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)



class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def sentence_block_embed(embed, x):
    """Computes sentence-level embedding representation from word-ids.

    :param embed: nn.Embedding() Module
    :param x: Tensor of batched word-ids
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    """
    batch, length = x.shape
    _, units = embed.weight.size()
    e = embed(x).transpose(1, 2).contiguous()
    assert (e.size() == (batch, units, length))
    return e


def seq_func(func, x, reconstruct_shape=True, pad_remover=None):
    """Change implicitly function's input x from ndim=3 to ndim=2

    :param func: function to be applied to input x
    :param x: Tensor of batched sentence level word features
    :param reconstruct_shape: boolean, if the output needs to be
    of the same shape as input x
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    or (batchsize x sentence_length, dimension)
    """
    batch, units, length = x.shape
    e = torch.transpose(x, 1, 2).contiguous().view(batch * length, units)
    if pad_remover:
        e = pad_remover.remove(e)
    e = func(e)
    if pad_remover:
        e = pad_remover.restore(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = torch.transpose(e.view((batch, length, out_units)), 1, 2).contiguous()
    assert (e.shape == (batch, out_units, length))
    return e


class LayerNormSent(LayerNorm):
    """Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length)."""

    def __init__(self, n_units, eps=1e-3):
        super(LayerNormSent, self).__init__(n_units, eps=eps)

    def forward(self, x):
        y = seq_func(super(LayerNormSent, self).forward, x)
        return y


class LinearSent(nn.Module):
    """Position-wise Linear Layer for sentence block. array of shape
    (batchsize, dimension, sentence_length)."""

    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSent, self).__init__()
        self.L = nn.Linear(input_dim, output_dim, bias=bias)
        # self.L.weight.data.uniform_(-3. / input_dim, 3. / input_dim)

        # Using Xavier Initialization
        # self.L.weight.data.uniform_(-math.sqrt(6.0 / (input_dim + output_dim)),
        #                             math.sqrt(6.0 / (input_dim + output_dim)))
        # LeCun Initialization
        self.L.weight.data.uniform_(-math.sqrt(3.0 / input_dim),
                                    math.sqrt(3.0 / input_dim))

        if bias:
            self.L.bias.data.fill_(0.)
        self.output_dim = output_dim

    def forward(self, x, pad_remover=None):
        output = seq_func(self.L, x, pad_remover=pad_remover)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, n_units, multi_heads=8, attention_dropout=0.1,
                 pos_attn=False):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = LinearSent(n_units,
                              n_units,
                              bias=False)
        self.W_K = LinearSent(n_units,
                              n_units,
                              bias=False)
        self.W_V = LinearSent(n_units,
                              n_units,
                              bias=False)
        self.finishing_linear_layer = LinearSent(n_units,
                                                 n_units,
                                                 bias=False)
        self.h = multi_heads
        self.pos_attn = pos_attn
        self.dropout = nn.Dropout(attention_dropout)

        # Rotary Position Embedding (RoPE) — precomputed for up to 2048 positions
        head_dim = n_units // multi_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(2048).float()
        freqs = torch.outer(t, inv_freq)              # (2048, head_dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)     # (2048, head_dim)
        self.register_buffer('rope_cos', emb.cos())
        self.register_buffer('rope_sin', emb.sin())

    @staticmethod
    def _rotate_half(x):
        """Rotate the second half of head_dim to implement RoPE."""
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def _apply_rope(self, x, seq_len):
        """Apply Rotary Position Embedding to x: (batch, heads, seq, head_dim)."""
        cos = self.rope_cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq, hd)
        sin = self.rope_sin[:seq_len].unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x, z=None, mask=None):
        h = self.h
        Q = self.W_Q(x)

        if not self.pos_attn:
            K, V = (self.W_K(x), self.W_V(x)) if z is None else (self.W_K(z), self.W_V(z))
        else:
            K, V = self.W_K(x), self.W_V(z)

        batch, n_units, n_queries = Q.shape
        _, _, n_keys = K.shape
        head_dim = n_units // h

        # Reshape: (batch, n_units, seq) → (batch, heads, seq, head_dim)
        Q = Q.view(batch, h, head_dim, n_queries).transpose(2, 3)
        K = K.view(batch, h, head_dim, n_keys).transpose(2, 3)
        V = V.view(batch, h, head_dim, n_keys).transpose(2, 3)

        # Apply Rotary Position Embedding to queries and keys
        Q = self._apply_rope(Q, n_queries)
        K = self._apply_rope(K, n_keys)

        # Expand bool mask: (batch, q, k) → (batch, 1, q, k) for head broadcasting
        attn_mask = mask.unsqueeze(1) if mask is not None else None

        # Flash Attention: fused kernel, O(n) memory, no explicit attention matrix
        dropout_p = self.dropout.p if self.training else 0.0
        C = F.scaled_dot_product_attention(Q, K, V,
                                           attn_mask=attn_mask,
                                           dropout_p=dropout_p)

        # Merge heads: (batch, heads, seq, head_dim) → (batch, n_units, seq)
        C = C.transpose(2, 3).contiguous().view(batch, n_units, n_queries)
        return self.finishing_linear_layer(C)


class FeedForwardLayer(nn.Module):
    """SwiGLU feed-forward sublayer: gate(SiLU) ⊗ up-projection → down-projection.

    Uses two parallel input projections (W_gate, W_up) and one output projection
    (W_2).  Hidden dim is scaled to 2/3 of n_hidden so total parameter count
    stays the same as the original two-matrix ReLU FFN.
    """

    def __init__(self, n_units, n_hidden, relu_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        # 3 matrices × (n_units × n_swiglu) ≈ 2 matrices × (n_units × n_hidden)
        n_swiglu = int(n_hidden * 2 / 3)
        self.W_gate = LinearSent(n_units, n_swiglu)
        self.W_up   = LinearSent(n_units, n_swiglu)
        self.W_2    = LinearSent(n_swiglu, n_units)
        self.dropout = nn.Dropout(relu_dropout)

    def forward(self, e, pad_remover=None):
        gate = F.silu(self.W_gate(e, pad_remover=pad_remover))
        up   = self.W_up(e,   pad_remover=pad_remover)
        e    = self.dropout(gate * up)
        return self.W_2(e, pad_remover=pad_remover)


class EncoderLayer(nn.Module):
    def __init__(self, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1, n_hidden=2048,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.ln_1 = LayerNormSent(n_units,
                                  eps=1e-3)
        self.self_attention = MultiHeadAttention(n_units,
                                                 multi_heads,
                                                 attention_dropout)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)
        self.ln_2 = LayerNormSent(n_units,
                                  eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units,
                                             n_hidden,
                                             relu_dropout)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)

    def forward(self, e, xx_mask, pad_remover=None):
        # e = self.ln_1(e)
        sub = self.self_attention(self.ln_1(e),
                                  mask=xx_mask)
        e = e + self.dropout1(sub)

        # e = self.ln_2(e)
        sub = self.feed_forward(self.ln_2(e),
                                pad_remover=pad_remover)
        e = e + self.dropout2(sub)
        return e


class DecoderLayer(nn.Module):
    def __init__(self, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1,
                 pos_attention=False, n_hidden=2048,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.pos_attention = pos_attention
        self.ln_1 = LayerNormSent(n_units,
                                  eps=1e-3)
        self.self_attention = MultiHeadAttention(n_units,
                                                 multi_heads,
                                                 attention_dropout)
        self.dropout1 = nn.Dropout(layer_prepostprocess_dropout)

        if pos_attention:
            pos_enc_block = Transformer.initialize_position_encoding(500,
                                                                     n_units)
            self.pos_enc_block = nn.Parameter(torch.FloatTensor(pos_enc_block),
                                              requires_grad=False)
            self.register_parameter("Position Encoding Block",
                                    self.pos_enc_block)

            self.ln_pos = LayerNormSent(n_units,
                                        eps=1e-3)
            self.pos_attention = MultiHeadAttention(n_units,
                                                    multi_heads,
                                                    attention_dropout,
                                                    pos_attn=True)
            self.dropout_pos = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_2 = LayerNormSent(n_units,
                                  eps=1e-3)
        self.source_attention = MultiHeadAttention(n_units,
                                                   multi_heads,
                                                   attention_dropout)
        self.dropout2 = nn.Dropout(layer_prepostprocess_dropout)

        self.ln_3 = LayerNormSent(n_units,
                                  eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units,
                                             n_hidden,
                                             relu_dropout)
        self.dropout3 = nn.Dropout(layer_prepostprocess_dropout)

    def forward(self, e, s, xy_mask, yy_mask, pad_remover):
        batch, units, length = e.shape

        # e = self.ln_1(e)
        sub = self.self_attention(self.ln_1(e),
                                  mask=yy_mask)
        e = e + self.dropout1(sub)

        if self.pos_attention:
            # e = self.ln_pos(e)
            p = self.pos_enc_block[:, :, :length]
            p = p.expand(batch, units, length)
            sub = self.pos_attention(p,
                                     self.ln_pos(e),
                                     mask=yy_mask)
            e = e + self.dropout_pos(sub)

        # e = self.ln_2(e)
        sub = self.source_attention(self.ln_2(e),
                                    s,
                                    mask=xy_mask)
        e = e + self.dropout2(sub)

        # e = self.ln_3(e)
        sub = self.feed_forward(self.ln_3(e),
                                pad_remover=pad_remover)
        e = e + self.dropout3(sub)
        return e


class Encoder(nn.Module):
    def __init__(self, n_layers, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1, n_hidden=2048,
                 attention_dropout=0.1, relu_dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = EncoderLayer(n_units,
                                 multi_heads,
                                 layer_prepostprocess_dropout,
                                 n_hidden,
                                 attention_dropout,
                                 relu_dropout)
            self.layers.append(layer)
        self.ln = LayerNormSent(n_units,
                                eps=1e-3)

    def forward(self, e, xx_mask, pad_remover):
        for layer in self.layers:
            e = layer(e,
                      xx_mask,
                      pad_remover)
        e = self.ln(e)
        return e


class Decoder(nn.Module):
    def __init__(self, n_layers, n_units, multi_heads=8,
                 layer_prepostprocess_dropout=0.1, pos_attention=False,
                 n_hidden=2048, attention_dropout=0.1,
                 relu_dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            layer = DecoderLayer(n_units,
                                 multi_heads,
                                 layer_prepostprocess_dropout,
                                 pos_attention,
                                 n_hidden,
                                 attention_dropout,
                                 relu_dropout)
            self.layers.append(layer)
        self.ln = LayerNormSent(n_units,
                                eps=1e-3)

    def forward(self, e, source, xy_mask, yy_mask, pad_remover):
        for layer in self.layers:
            e = layer(e,
                      source,
                      xy_mask,
                      yy_mask,
                      pad_remover)
        e = self.ln(e)
        return e


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embed_word = ScaledEmbedding(config.n_vocab,
                                          config.n_units,
                                          padding_idx=0)
        self.embed_dropout = nn.Dropout(config.dropout)
        self.n_hidden = config.n_units * 4
        self.encoder = Encoder(config.layers,
                               config.n_units,
                               config.multi_heads,
                               config.layer_prepostprocess_dropout,
                               self.n_hidden,
                               config.attention_dropout,
                               config.relu_dropout)

        self.decoder = Decoder(config.layers,
                               config.n_units,
                               config.multi_heads,
                               config.layer_prepostprocess_dropout,
                               config.pos_attention,
                               self.n_hidden,
                               config.attention_dropout,
                               config.relu_dropout)

        if config.embed_position:
            self.embed_pos = nn.Embedding(config.max_length,
                                          config.n_units,
                                          padding_idx=0)

        if config.tied:
            self.affine = self.tied_linear
        else:
            self.affine = nn.Linear(config.n_units,
                                    config.n_vocab,
                                    bias=True)

        self.n_target_vocab = config.n_vocab
        self.dropout = config.dropout
        self.label_smoothing = config.label_smoothing
        self.scale_emb = config.n_units ** 0.5

        # Positional encoding is handled by RoPE inside MultiHeadAttention.
        # The sinusoidal pos_enc_block is no longer added to embeddings.

    @staticmethod
    def initialize_position_encoding(length, emb_dim):
        channels = emb_dim
        position = np.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (np.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * np.exp(np.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.reshape(signal, [1, length, channels])
        pos_enc_block = np.transpose(signal, (0, 2, 1))
        return pos_enc_block

    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        # No sinusoidal offset — position information is injected via RoPE in attention.

        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(self.embed_pos,
                                              np.broadcast_to(np.arange(length).astype('i')[None, :],
                                                              block.shape))
        emb_block = self.embed_dropout(emb_block)
        return emb_block

    def make_attention_mask(self, source_block, target_block):
        mask = (target_block[:, None, :] >= 1) & \
               (source_block[:, :, None] >= 1)
        # (batch, source_length, target_length)
        return mask

    def make_history_mask(self, block):
        batch, length = block.shape
        arange = np.arange(length)
        history_mask = (arange[None,] <= arange[:, None])[None,]
        history_mask = np.broadcast_to(history_mask,
                                       (batch, length, length))
        history_mask = Variable(torch.BoolTensor(history_mask.astype(bool)),
                                requires_grad=False)
        if utils.LONG_TYPE == torch.cuda.LongTensor:
            history_mask = history_mask.cuda()
        return history_mask

    def tied_linear(self, h):
        return F.linear(h, self.embed_word.weight)

    def output(self, h):
        return self.affine(h)

    def output_and_loss(self, h_block, t_block):
        batch, units, length = h_block.shape
        # shape : (batch * sequence_length, num_classes)
        logits_flat = seq_func(self.affine,
                               h_block,
                               reconstruct_shape=False)
        rebatch, _ = logits_flat.shape
        concat_t_block = t_block.view(rebatch)
        weights = (concat_t_block >= 1).float()
        n_correct, n_total = utils.accuracy(logits_flat,
                                            concat_t_block,
                                            ignore_index=0)

        # shape : (batch * sequence_length, num_classes)
        log_probs_flat = F.log_softmax(logits_flat,
                                       dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = t_block.view(-1, 1).long()

        if self.label_smoothing is not None and self.label_smoothing > 0.0:
            num_classes = logits_flat.size(-1)
            smoothing_value = self.label_smoothing / (num_classes - 1)
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = input_like(log_probs_flat,
                                         smoothing_value)
            smoothed_targets = one_hot_targets.scatter_(-1,
                                                        targets_flat,
                                                        1.0 - self.label_smoothing)
            negative_log_likelihood_flat = - log_probs_flat * smoothed_targets
            negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1,
                                                                            keepdim=True)
        else:

            negative_log_likelihood_flat = - torch.gather(log_probs_flat,
                                                          dim=1,
                                                          index=targets_flat)

        # shape : (batch, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(rebatch)
        negative_log_likelihood = negative_log_likelihood * weights
        # shape : (batch_size,)
        loss = negative_log_likelihood.sum() / (weights.sum() + 1e-13)
        stats = utils.Statistics(loss=utils.to_cpu(loss) * n_total,
                                 n_correct=utils.to_cpu(n_correct),
                                 n_words=n_total)
        return loss, stats

    def forward(self, x_block, y_in_block, y_out_block, get_prediction=False,
                z_blocks=None):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        if z_blocks is None:
            ex_block = self.make_input_embedding(self.embed_word,
                                                 x_block)
            xx_mask = self.make_attention_mask(x_block,
                                               x_block)
            xpad_obj = PadRemover(x_block >= preprocess.Vocab_Pad.PAD)
            # Encode Sources
            z_blocks = self.encoder(ex_block,
                                    xx_mask,
                                    xpad_obj)
            # (batch, n_units, x_length)

        ey_block = self.make_input_embedding(self.embed_word,
                                             y_in_block)
        # Make Masks
        xy_mask = self.make_attention_mask(y_in_block,
                                           x_block)
        yy_mask = self.make_attention_mask(y_in_block,
                                           y_in_block)
        yy_mask *= self.make_history_mask(y_in_block)

        # Create PadRemover objects
        ypad_obj = PadRemover(y_in_block >= preprocess.Vocab_Pad.PAD)

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(ey_block,
                               z_blocks,
                               xy_mask,
                               yy_mask,
                               ypad_obj)
        # (batch, n_units, y_length)

        if get_prediction:
            return self.output(h_block[:, :, -1]), z_blocks
        else:
            return self.output_and_loss(h_block,
                                        y_out_block)

    def translate(self, x_block, max_length=50, beam=5, alpha=0.6):
        if beam:
            obj = decoding.BeamSearch(beam_size=beam,
                                             max_len=max_length,
                                             alpha=alpha)
            id_list, score = obj.generate_output(self,
                                                 x_block)
            return id_list, score
        else:
            obj = decoding.GreedySearch(max_len=max_length)
            id_list = obj.generate_output(self,
                                          x_block)
            return id_list, None
