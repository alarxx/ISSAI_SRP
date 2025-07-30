import torch

x = torch.tensor([
    [[1, 2, 0], 
     [3, 4, 0]], # Batch 0

    [[5, 6, 0], 
     [7, 8, 0]], # Batch 1

    [[6, 7, 0], 
     [8, 9, 0]], # Batch 2
]) 
print(x)
print(x.shape)

# or same, but using permute
x_t = x.transpose(1, 0)
print(x_t)
print(x_t.shape)  # ось 0 и 1 поменялись местами

x_p = x.permute(1, 0, 2)
print(x_p)
print(x_p.shape)  