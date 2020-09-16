import torch
import torch.nn as nn
from torch.autograd import Variable

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class dSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))

class mdSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        input += 4
        scale = 3
        return scale * (torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input))))
