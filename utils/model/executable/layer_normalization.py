import torch
from torch import nn

# 2 последовательности длиной 2 токена, embedding dim = 3
x = torch.tensor([
    [[2, 3, 5], # batch 1
     [2, 3, 5]],

    [[2, 3, 5], # batch 2
     [2, 3, 5]],
], dtype=torch.float32)
print("x:", x)
print("x.shape:", x.shape)

ln = nn.LayerNorm(3) # -> [[[-1.0690, -0.2673,  1.3363], ...], ...]

y = ln(x)
print()
print("y:", y)
print("y.shape:", y.shape)

# Проверим среднее и std по последней оси
print()
print("mean:", y.mean(-1))  # считает mean по последнему измерению
print("std:", y.std(-1))   # по каждому (b, t)

# (x - m) / r^2
# (2 - 3.3) / 1.2247 = -1.06
# (3 - 3.3) / 1.2247 = -0.27
# (5 - 3.3) / 1.2247 = 1.38
