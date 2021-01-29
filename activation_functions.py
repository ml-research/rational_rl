import torch
import torch.nn as nn
from torch.autograd import Variable

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input.float())


class dSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(input.float()) * (1 + input * (1 - torch.sigmoid(input.float())))
