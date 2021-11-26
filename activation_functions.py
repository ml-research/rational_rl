import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.functional import relu

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


class PELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.FloatTensor([1.]))
        self.b = nn.Parameter(torch.FloatTensor([1.]))
        self.c = nn.Parameter(torch.FloatTensor([1.]))

    def forward(self, x):
        return self.c * relu(x.float()) + self.a * (torch.exp(-relu(-x.float())/self.b) - 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pelu = PELU()
    inp = torch.arange(-3, 3, 0.01)
    plt.grid()
    plt.plot(inp.detach().numpy(), pelu(inp).detach().numpy())
    plt.show()
