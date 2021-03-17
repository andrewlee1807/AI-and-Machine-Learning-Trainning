import torch
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init

dtype = torch.FloatTensor
input_size, hidden_size, output_size = 7, 6, 1
epochs = 300
seq_length = 20
lr = 0.1
data_time_steps = np.linspace(2,10, seq_length +1 )
data = np.sin(data_time_steps)
data.resize((seq_length + 1, 1))

x = Variable(torch.Tensor(data[:-1]).type(dtype), requires_grad = False)
y = Variable(torch.Tensor(data[1:]).type(dtype), requires_grad = False)




