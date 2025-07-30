import torch
from torch import nn, Tensor
import math


# --- Positional Encoding ---

class PositionalEncoding(nn.Module):
    """
        https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
        PositionalEncoding module injects some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings so that the two can be summed.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# --- Transformer (Decoder) Model ---

class MyTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # В конце нужно умножать на тот же embedding, а не на новый Linear
        # self.output_proj = nn.Linear(d_model, vocab_size)
        # Weight tying
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.embedding(tgt)
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.transformer_decoder(tgt_emb, memory=None, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask)
        return self.output_proj(output)