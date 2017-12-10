"""
Pytorch tutorial
http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
"""
from __future__ import print_function
import torch
import numpy as np

# Part 1
x = torch.rand(5, 3)
y = torch.rand(5, 3)
#print(y.size())

# New matrix from addition
z = x + y
#print(z)

# Or in-place addition
y = y.add_(x)
#print(y)

# numpy like indexing
#print(x[:, 1]) # first column

# bridge to numpy
a = torch.ones(5)
#print(a, a.numpy())

b = np.ones(5)
#print(torch.from_numpy(b))

if torch.cuda.is_available():
    print("CUDA")

# Part 2
# http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
from torch.autograd import Variable
x_a = Variable(torch.ones(2,2), requires_grad = True)
#print(x_a)
y_a = x_a + 2
#print(y_a.grad_fn) # becuz created by an operation, y holds a grad function
z_a = y_a * y_a * 3
out = z_a.mean()
#print(out)

out.backward() # backprop
#print(x_a.grad)

def example_autograd():
    x = torch.randn(3)
    x = Variable(x, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    print(y)
    gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
    y.backward(gradients)
    print(x.grad)


example_autograd()
