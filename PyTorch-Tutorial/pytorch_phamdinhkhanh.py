import torch

x = torch.empty(5, 3)
y = torch.ones((5, 3), dtype=torch.int)
z = torch.rand(3, 4)
z = torch.randint(1, 10, (5, 2))
z = torch.randn_like(x, dtype=torch.float32)
a = torch.Tensor([[[2, 3, 4, 5], [23, 4, 4, 6]]])
print(x)
print(y)
print(z)
print(z.dtype)
print(a)
print(a.shape)

#  Operations
print("===Operations===")
x = torch.ones(3, 3)
y = torch.randint(1, 10, (3, 3), dtype=torch.float)
print(x)
print(y)

z = torch.empty(3, 3)
torch.add(x, y, out=z)
print(x + y)
print(torch.add(x, y))
print(z)

print(x @ y)
print(torch.matmul(x, y))
print(torch.mm(x, y))
