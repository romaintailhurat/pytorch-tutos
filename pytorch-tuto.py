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
print(y.size())

# New matrix from addition
z = x + y
print(z)

# Or in-place addition
y = y.add_(x)
print(y)

# numpy like indexing
print(x[:, 1]) # first column

# bridge to numpy
a = torch.ones(5)
print(a, a.numpy())

b = np.ones(5)
print(torch.from_numpy(b))

if torch.cuda.is_available():
    print("CUDA")

# Part 2
# http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
