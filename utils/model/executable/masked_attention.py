"""
    Masked-Attention = Attention + Mask
    Чтобы замаскировать следующие слова для каждого предсказания,
    которые происходят за один проход,
    мы можем прибавить диагональную матрицу со значениями -inf выше диагонали
    (диагональ не включительно).
"""

import torch


# --- About float('-inf') ---

x = float('inf')
y = float('-inf')

print(x + 5) # inf
print(y + 5) # -inf

print(1 / float('-inf'))  # 0.0
print(float('inf') / float('inf'))  # nan

a = torch.tensor([0.0, float('-inf'), float('inf')])
print(a) # tensor([0., -inf, inf])

b = torch.exp(torch.tensor(float('-inf')))
print(b)  # tensor(0.)


# --- Generating Mask ---

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

mask = generate_causal_mask(5)
print(mask)


# --- Masked Attention ---

attn = torch.tensor([
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 5],
], dtype=torch.float32)

attn += mask

print(attn)
