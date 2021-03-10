import torch

print(torch.cuda.get_device_capability())
print(torch.cuda.get_device_name())

import numpy as np

tensor = torch.FloatTensor(np.random.rand(7))
scalar = torch.rand([3, 4])

from sklearn.datasets import load_boston

boston = load_boston()
boston_tenser = torch.from_numpy(boston.data)
boston_tenser_te = torch.Tensor(boston.data)


#  CREATE A FIRST NEURAL NETWORK
def first_neural_network():
    # Step 1
    import torch
    import torch.nn as nn
    torch.cuda.device(0)

    # Step 2
    # Defining input size, hidden layer size, output size and batch size respectively
    n_in, n_h, n_out, batch_size = 10, 5, 1, 10

    #  Step 3
    # Create dummy input and target tensors (data)
    x = torch.randn(batch_size, n_in).cuda()
    y = torch.tensor([[1.0], [0.0], [0.0],
                      [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]]).cuda()

    # y = torch.tensor(np.random.randint(2, size=(10)))

    # Step 4
    # Create a model
    model = nn.Sequential(nn.Linear(n_in, n_h),
                          nn.ReLU(),
                          nn.Linear(n_h, n_out),
                          nn.Sigmoid()).cuda()

    # Step 5
    # Construct the loss function
    criterion = torch.nn.MSELoss().cuda()
    # Construct the optimizer (Stochastic Gradient Descent in this case)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Step 6
    # Gradient Descent
    for epoch in range(50):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print('epoch: ', epoch, ' loss: ', loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward()

        # Update the parameters
        optimizer.step()


#  Download data
# import torchvision
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                           download=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

sns.set_style(style='whitegrid')
plt.rcParams["patch.force_edgecolor"] = True

#  Create a single training set with the available data set as shown below −
m = 2  # slope
c = 3  # interceptm = 2 # slope
c = 3  # intercept
x = np.random.rand(256)

noise = np.random.randn(256) / 4

y = x * m + c + noise

df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x='x', y='y', data=df)


# Implement linear regression with PyTorch libraries as mentioned below
def linear_regression_model():
    import torch
    import torch.nn as nn
    from torch.autograd import Variable

    x_train = x.reshape(-1, 1).astype('float32')
    y_train = y.reshape(-1, 1).astype('float32')

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim).cuda()

        def forward(self, x):
            out = self.linear(x)
            return out

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    # input_dim, output_dim(1, 1)
    model = LinearRegressionModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    [w, b] = model.parameters()

    def get_para_value():
        return w.data[0][0], b.data[0]

    def plot_current_fit(title=""):
        plt.figure(figsize=(12, 4))
        plt.title(title)
        plt.scatter(x, y, s=8)
        w1 = w.data[0][0].cpu().numpy()
        b1 = b.data[0].cpu().numpy()
        x1 = np.array([0., 1.])
        y1 = x1 * w1 + b1
        plt.plot(x1, y1, 'r', label='Current Fit ({:.3f}, {:.3f})'.format(w1, b1))
        plt.xlabel('x (input)')
        plt.ylabel('y (target)')
        plt.legend()
        plt.show()

    plot_current_fit('Before training')


def convolutional_neural_network():
    # Step 1. Import the necessary packages for creating a simple neural network.
    from torch.autograd import Variable
    import torch.nn.functional as F
    #  Step 2. Create a class with batch representation of convolutional neural network.
    #  Our batch shape for input x is with dimension of (3, 32, 32).
    class SimpleCNN(torch.nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            #       Input Channels = 3, Output Channels = 18
            self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
            self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # 4608 input features, 64 output features (see sizing flow below)
            self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
            # 64 input features, 10 output features for our 10 defined classes
            self.fc2 = torch.nn.Linear(64, 10)

        # Compute the activation of the first convolution size changes from (3, 32, 32) to (18, 32, 32).
        #
        # Size of the dimension changes from (18, 32, 32) to (18, 16, 16).
        # Reshape data dimension of the input layer of the neural net due to which size changes from
        # (18, 16, 16) to (1, 4608).
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(-1, 18 * 16 * 16)  # Recall that -1 infers this dimension from the other given dimension.
            x = F.relu(self.fc1(x))
            # Computes the second fully connected layer (activation applied later)
            # Size changes from (1, 64) to (1, 10)
            x = self.fc2(x)
            return x
