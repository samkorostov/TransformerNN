import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # scale parameter
        self.bias = nn.Parameter(torch.zeros(1))  # shift parameter

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)  # W1 and b1
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(d_ff, d_model)  # W2 and b2

    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        self.fc_q = nn.Linear(d_model, d_model)  # W_q
        self.fc_k = nn.Linear(d_model, d_model)  # W_k
        self.fc_v = nn.Linear(d_model, d_model)  # W_v

        self.fc_o = nn.Linear(d_model, d_model)  # W_o
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.size(-1)

        attn = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(attn, dim=-1)  # (Batch, h, Seq, Seq)

        if dropout is not None:
            attn = dropout(attn)

        return (attn @ value), attn

    def forward(self, q, k, v, mask=None):
        query = self.fc_q(q)  # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        key = self.fc_k(k)  # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        value = self.fc_v(v)  # (Batch, Seq, d_model) -> (Batch, Seq, d_model)

        # (Batch, Seq, d_model) -> (Batch, h, Seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attn = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq, d_k) -> (Batch, Seq, h, d_k) -> (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        return self.fc_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        return self.residual_connection[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        '''
        :param x: Input to decoder
        :param memory: Input from encoder
        :param src_mask: Mask coming from encoder
        :param tgt_mask: Mask coming from decoder
        '''

        m = memory
        # Self-Attention
        x = self.residual_connection[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Cross-Attention
        x = self.residual_connection[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.residual_connection[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ProjectionLayer, self).__init__()

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, vocab_size)
        return torch.log_softmax(self.fc(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x)


def make_model(src_vocab_size,
               tgt_vocab_size,
               src_seq_len,
               tgt_seq_len,
               d_model=512,
               N=6,
               h=8,
               dropout=0.1,
               d_ff=2048):
    # Create embeddings
    src_embed = Embeddings(d_model, src_vocab_size)
    tgt_embed = Embeddings(d_model, tgt_vocab_size)

    # Create positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder and decoder layers
    encoder_layers = []
    for _ in range(N):
        encoder_self_attn = MultiHeadAttention(d_model, h, dropout)
        encoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        encoder_layers.append(EncoderLayer(encoder_self_attn, encoder_feed_forward, dropout))

    decoder_layers = []
    for _ in range(N):
        decoder_self_attn = MultiHeadAttention(d_model, h, dropout)
        decoder_src_attn = MultiHeadAttention(d_model, h, dropout)
        decoder_feed_forward = FeedForward(d_model, d_ff, dropout)
        decoder_layers.append(DecoderLayer(decoder_self_attn, decoder_src_attn, decoder_feed_forward, dropout))

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_layers))
    decoder = Decoder(nn.ModuleList(decoder_layers))

    # Create projection layer
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer model
    model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    # Initialize model parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Return model
    return model


