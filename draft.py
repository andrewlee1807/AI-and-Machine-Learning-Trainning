import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2), requires_grad=True)
y = x + 2
z = y * y * 2
out = z.mean()

out.backward()
#  Dynamic computation Graph
print(y.grad_fn)     # The Function that create the Variable y



import torch
from torch.autograd import Variable

dtype = torch.FloatTensor
N, D_in, H, D_out = 64, 1000, 100, 10

# Dataset and label
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

w1 = torch.randn((D_in, H),requires_grad=True).type(dtype)
w2 = torch.randn((H, D_out), requires_grad=True).type(dtype)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)   # combine RELU inside this equation
    # loss = (y_pred - y).pow(2).sum()
    loss = torch.sum((y_pred - y).pow(2))
    print(t, loss.item())

    loss.backward()

    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    w1.grad.data.zero_()
    w2.grad.data.zero_()



