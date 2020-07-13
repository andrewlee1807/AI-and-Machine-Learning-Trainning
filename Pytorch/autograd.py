import torch
x = torch.ones(2,2, requires_grad=True)
print(x)
y = x + 2
print(y)
z = y*y**3
out = z.mean()
print(out)
print(z)

out.backward()
print(x.grad)

print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())