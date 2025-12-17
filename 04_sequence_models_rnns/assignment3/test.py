import torch

a = torch.randn(20, 4, 64)
print(a)
b = a.split(1, dim=0)
for chunk in b:
    print(chunk.shape)