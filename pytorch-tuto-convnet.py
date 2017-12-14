# Neural nets
# http://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()

print(">> Learnable parameters of the NN")
params = list(net.parameters())
print(len(params))
print(params[0].size())

print(">> Testing with a random input")
input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)
print(out)

target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()
loss = criterion(out, target)
print(">> Loss :\n", loss)

print(">> ")
net.zero_grad()
print("bias before", net.conv1.bias.grad)
loss.backward()
print("bias after", net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
