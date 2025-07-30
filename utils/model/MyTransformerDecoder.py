import torch
from torch import nn, Tensor
import math


# --- Absolute Positional Encoding ---

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


# --- GPTDecoderBlock ---

# Block в модели будет повторяться несколько раз
# Modern Pre-Layer Normalization
class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        """
            Args:
                dropout - drop probability; keep_probability = 1 - drop_probability
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) # normalize row by columns <-->
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # MHA
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=attn_mask,  # shape: (T, T)
            key_padding_mask=key_padding_mask  # shape: (B, T), bool
        )
        x = x + self.dropout1(attn_out) # Skip connection
        # MLP
        x_norm = self.ln2(x)
        x = x + self.mlp(x_norm) # Skip connection
        return x


# --- Generating Mask for Masked Self-Attention ---

def generate_causal_mask(seq_len, device="cpu"):
    """
        Args:
            seq_len: number of tokens of input sequence of text
    """
    full = torch.full((seq_len, seq_len), float("-inf"))
    # The upper triangular part of a matrix retained, and the other elements are set to 0.
    # diagonal=0 retains the main diagonal, diagonal=1 goes up and set it to 0.
    triu = torch.triu(full, diagonal=1).to(device)
    return triu


# --- MyTransformerDecoder ---

class MyTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_ff=2048, dropout=0.1, max_len=5000, pad_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.token_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_token_id)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        self.blocks = nn.ModuleList([
            GPTDecoderBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        # В конце нужно умножать на тот же embedding, а не на новый Linear
        # self.output_proj = nn.Linear(d_model, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embed.weight

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (B, T) — токены
        attention_mask: (B, T) — 1 для валидных токенов, 0 для паддинга (можно None)
        """
        B, T = input_ids.size()
        device = input_ids.device

        # embedding
        x = self.token_embed(input_ids)  # (B, T, D)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)  # (T, B, D) -> PE -> (B, T, D)
        # transpose(0, 1) = permute(1, 0, 2)

        # attention mask (causal, triangular)
        attn_mask = generate_causal_mask(T, device)  # shape: (T, T)

        # padding mask (different length of sequences)
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # invert: 1 -> False, 0 -> True
        else:
            key_padding_mask = (input_ids == self.pad_token_id)  # shape: (B, T), bool

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)
        return self.lm_head(x)  # shape: (B, T, vocab_size)
