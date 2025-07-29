"""
    Предложения могут быть разной длины и чтобы обучать разные предложения в батче
    мы должны поместить разные предложения в квадратный тензор,
    но тогда их длина должна быть одинаковой.
    Чтобы решить это мы просто дополняем более короткие предложения токенами [PAD],
    Но в таком случае наша модель не должна видеть эти токены,
    поэтому мы маскируем их в Attention матрице.
"""

import torch

# Q K attention square matrix
attn = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.float32)

attn = attn.unsqueeze(0) # like 1 batch

print(attn.shape) # (1, 3, 3) in (B, R, C) represents (B, T, T), where T is sequence length

# For example, last token is [PAD]
attention_mask = torch.tensor([
    [True, True, False] # for 1 batch
])
# No big reason to convert it to long, just make it fault tolerant with .bool()
attention_mask = attention_mask.long() # [ [1, 1, 0] ]
print("attention_mask:", attention_mask)

# Negation of attention mask, whick masks [PAD] tokens
# Again, no big reason to negate if we could make right comparison in the beggining, (input_ids == pad_token_id)
key_padding_mask = ~attention_mask.bool() # without .bool() negation(~) may not work
# [ [False, False, True] ]
print("key_padding_mask:", key_padding_mask)

key_padding_mask = key_padding_mask.unsqueeze(1).to(torch.bool)
# [ [[False, False, True]] ] # B, R, C
print(key_padding_mask.shape) # (1, 1, 3)

attn = attn.masked_fill(
    key_padding_mask,  # (B, 1, T_k)
    float('-inf')
)

print(attn) # masked columns
