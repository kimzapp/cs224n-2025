import torch

a = torch.tensor([[[1,2,3], [4,5,6]], [[7,8,9], [10, 11, 12]]])
print(a.shape)
print(a)
b = a.permute(dims=(1, 0, 2)).flatten(start_dim=1)
print(b)
print(b.shape)