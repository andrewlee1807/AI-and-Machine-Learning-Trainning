import torch

x = torch.tensor([[3, 1, 2], [4, 1, 7]])
print(x)
# Transpose the Tensor
print(x.t())

# Contiguous tensors
x.is_contiguous()   # Check ordering storage of Tensor



