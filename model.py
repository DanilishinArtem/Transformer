import torch
import torch.nn as nn
import math


# class of the embedding layer
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# class of the positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of the shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # saving at the buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplier
        self.beta = nn.Parameter(torch.zeros(1)) # shift

    def forwardd(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super.__innit__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, d_ff) -> (Batch, Seq, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super.__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % self.num_heads == 0, 'd_model is not divisible by num_heads'

        self.d_k = d_model // self.num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_score = None

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, dropout: nn.Dropout) -> torch.Tensor:
        d_k = query.shape[-1]
        # (Batch, Num_heads, Seq, d_k) -> (Batch, Num_heads, Seq, Seq)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, Num_heads, Seq, Seq)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return torch.matmul(attention_scores, value), attention_scores
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (Batch, Seq, d_model) -> (Batch, Num_heads, Seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, Num_heads, Seq, d_k) -> (Batch, Seq, Num_heads, d_k) -> (Batch, Seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

        # (Batch, Seq, d_model) -> (Batch, Seq, d_model)
        return self.w_o(x)


class ResidualConnectionBlock(nn.Module):

    def __init__(self, dropout: float) -> None:
        super.__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: nn.Module, feed_forward_block: nn.Module, dropout: float) -> None:
        super.__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super.__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super.__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnectionBlock(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super.__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (Batch, Seq, d_model) -> (Batch, Seq, vocab_size)
        return torch.log_softmax(self.projection(x), dim = -1)


class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_position: PositionalEncoding, tgt_position: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_position(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Creating embeding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, encoder_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer
    transformer = Transformer(encoder, decoder, src_pos, tgt_pos, src_embed, tgt_embed, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

    