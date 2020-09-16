import torch
import torch.nn as nn
import torch.nn.functional as F
from pau_torch.pade_activation_unit import PAU
from utils import sepprint
from activation_functions import SiLU, dSiLU
from physt import h1 as hist1

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    sepprint("\nUsing CUDA on " + torch.cuda.get_device_name(0) + '\n')
else:
    sepprint("\nNot using CUDA\n")


class Network(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, activation_function,
                 freeze_pau, loaded_act_f=None, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        self._h1 = nn.Conv2d(n_input, 32, kernel_size=8, stride=4)
        self._h2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self._h3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self._h4 = nn.Linear(3136, self.n_features)
        self._h5 = nn.Linear(self.n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h4.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h5.weight,
                                gain=nn.init.calculate_gain('linear'))

        if activation_function == "rpau":
            if loaded_act_f is not None:
                self.act_func1 = loaded_act_f[0]
                self.act_func2 = self.act_func1
                self.act_func3 = self.act_func1
                self.act_func4 = self.act_func1
            else:
                self.act_func1 = PAU(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func2 = self.act_func1
                self.act_func3 = self.act_func1
                self.act_func4 = self.act_func1
        elif activation_function == "paus":
            if loaded_act_f is not None:
                self.act_func1 = loaded_act_f[0]
                self.act_func2 = loaded_act_f[1]
                self.act_func3 = loaded_act_f[2]
                self.act_func4 = loaded_act_f[3]
            else:
                self.act_func1 = PAU(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func2 = PAU(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func3 = PAU(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func4 = PAU(cuda=USE_CUDA).requires_grad_(not freeze_pau)
        elif activation_function == "relu":
            self.act_func1 = F.relu
            self.act_func2 = self.act_func1
            self.act_func3 = self.act_func1
            self.act_func4 = self.act_func1
        elif activation_function == "lrelu":
            self.act_func1 = F.leaky_relu
            self.act_func2 = self.act_func1
            self.act_func3 = self.act_func1
            self.act_func4 = self.act_func1
            # self.shared_w =
        elif activation_function == "silu":
            self.act_func1 = SiLU()
            self.act_func2 = self.act_func1
            self.act_func3 = self.act_func1
            self.act_func4 = self.act_func1
        elif activation_function == "dsilu":
            self.act_func1 = dSiLU()
            self.act_func2 = self.act_func1
            self.act_func3 = self.act_func1
            self.act_func4 = self.act_func1
        elif activation_function == "d+silu":
            self.act_func1 = SiLU()
            self.act_func2 = self.act_func1
            self.act_func3 = dSiLU()
            self.act_func4 = self.act_func3



        # self.set_hists()

    def forward(self, state, action=None):
        x1 = self._h1(state.float() / 255.)
        h = self.act_func1(x1)
        x2 = self._h2(h)
        h = self.act_func2(x2)
        x3 = self._h3(h)
        h = self.act_func3(x3)
        x4 = self._h4(h.view(-1, 3136))
        h = self.act_func4(x4)
        q = self._h5(h)
        if hasattr(self, 'inp1'):
            self.inp1.fill_n(x1.detach().cpu().numpy())
            self.inp2.fill_n(x2.detach().cpu().numpy())
            self.inp3.fill_n(x3.detach().cpu().numpy())
            self.inp4.fill_n(x4.detach().cpu().numpy())
            x1.to('cuda:0'); x2.to('cuda:0'); x3.to('cuda:0'); x4.to('cuda:0')
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted

    def set_hists(self):
        self.inp1 = hist1(None, "fixed_width", bin_width=0.1, adaptive=True)
        self.inp2 = hist1(None, "fixed_width", bin_width=0.1, adaptive=True)
        self.inp3 = hist1(None, "fixed_width", bin_width=0.1, adaptive=True)
        self.inp4 = hist1(None, "fixed_width", bin_width=0.1, adaptive=True)
