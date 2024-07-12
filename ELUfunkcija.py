import torch
import torch.nn as nn
import math

class MyGelu(nn.Module):
    def __init__(self):
        super(MyGelu, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / torch.pi) * (x + 0.044715 * x ** 3)))