import torch
import torch.nn as nn
import torch.nn.functional as F
from rational.torch import Rational, EmbeddedRational
from utils import sepprint
from activation_functions import SiLU, dSiLU

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    sepprint("\nUsing CUDA on " + torch.cuda.get_device_name(0) + '\n')
else:
    sepprint("\nNot using CUDA\n")


class Network(nn.Module):
    n_features = 512

    def __init__(self, input_shape, output_shape, activation_function,
                 freeze_pau=False, loaded_act_f=None, **kwargs):
        super().__init__()

        n_input = input_shape[0]
        n_output = output_shape[0]

        if activation_function == "embrat":
            print("Using BatchNorm because of Embedded Rationals")
            self._h1 = nn.Sequential(nn.Conv2d(n_input, 32, kernel_size=8, stride=4), nn.BatchNorm2d(32, affine=False))
            self._h2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.BatchNorm2d(64, affine=False))
            self._h3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.BatchNorm2d(64, affine=False))
            self._h4 = nn.Linear(3136, self.n_features)
            self._h5 = nn.Linear(self.n_features, n_output)

            nn.init.xavier_uniform_(self._h1[0].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h2[0].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h3[0].weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h4.weight,
                                    gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self._h5.weight,
                                    gain=nn.init.calculate_gain('linear'))
        else:
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

        if activation_function == "recrat":
            if loaded_act_f is not None:
                self.act_func1 = loaded_act_f[0]
                self.act_func2 = self.act_func1
                self.act_func3 = self.act_func1
                self.act_func4 = self.act_func1
            else:
                self.act_func1 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func2 = self.act_func1
                self.act_func3 = self.act_func1
                self.act_func4 = self.act_func1
        elif activation_function == "embrat":
            self.act_func1 = EmbeddedRational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
            self.act_func2 = EmbeddedRational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
            self.act_func3 = EmbeddedRational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
            self.act_func4 = EmbeddedRational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
        elif activation_function == "rat":
            if loaded_act_f is not None:
                self.act_func1 = loaded_act_f[0]
                self.act_func2 = loaded_act_f[1]
                self.act_func3 = loaded_act_f[2]
                self.act_func4 = loaded_act_f[3]
            else:
                self.act_func1 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func2 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func3 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
                self.act_func4 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
        elif activation_function == "r2r2":
            self.act_func1 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
            self.act_func2 = self.act_func1
            self.act_func3 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
            self.act_func4 = self.act_func3
        elif activation_function == "r3r":
            self.act_func1 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
            self.act_func2 = self.act_func1
            self.act_func3 = self.act_func1
            self.act_func4 = Rational(cuda=USE_CUDA).requires_grad_(not freeze_pau)
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


    def forward(self, state, action=None):
        x1 = self._h1(state.float() / 255.)
        if x1.isnan().any():
            import ipdb; ipdb.set_trace()
        h = self.act_func1(x1)
        if h.isnan().any():
            import ipdb; ipdb.set_trace()
        x2 = self._h2(h)
        if x2.isnan().any():
            import ipdb; ipdb.set_trace()
        h = self.act_func2(x2)
        if h.isnan().any():
            import ipdb; ipdb.set_trace()
        x3 = self._h3(h)
        if x3.isnan().any():
            import ipdb; ipdb.set_trace()
        h = self.act_func3(x3)
        if h.isnan().any():
            import ipdb; ipdb.set_trace()
        x4 = self._h4(h.view(-1, 3136))
        if x4.isnan().any():
            import ipdb; ipdb.set_trace()
        h = self.act_func4(x4)
        if h.isnan().any():
            import ipdb; ipdb.set_trace()
        q = self._h5(h)
        if q.isnan().any():
            import ipdb; ipdb.set_trace()
        if action is None:
            return q
        else:
            q_acted = torch.squeeze(q.gather(1, action.long()))

            return q_acted

    def hasrecurrentrational(self):
        return type(self.act_func1) == Rational and \
                self.act_func1 == self.act_func2

    def input_retrieve_mode(self, max_saves=1000):
        if self.hasrecurrentrational():
            self.act_func1.input_retrieve_mode(max_saves=max_saves)
        else:
            self.act_func1.input_retrieve_mode(max_saves=max_saves)
            self.act_func2.input_retrieve_mode(max_saves=max_saves)
            self.act_func3.input_retrieve_mode(max_saves=max_saves)
            self.act_func4.input_retrieve_mode(max_saves=max_saves)

    def show(self, score_dict=None, save=False, args=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        if self.hasrecurrentrational():
            act_fs = [self.act_func1]
            fig, axes = plt.subplots(1, 1, figsize=(4.5, 3))
            axes = [axes]
            lab = "R.RAF"
            if score_dict is not None:
                lab += f" ({score_dict['rat']})"
        else:
            act_fs = [self.act_func1, self.act_func2, self.act_func3, self.act_func4]
            fig, axes = plt.subplots(1, 4, figsize=(15, 4))
            lab = "RAF"
        acts_contents = []
        for act in act_fs:
            acts_contents.append(act.show(display=False))
        for content, ax in zip(acts_contents, axes):
            if content["hist"] is not None:
                hist = content["hist"]
                ax2 = ax.twinx()
                ax2.set_yticks([])
                grey_color = (0.5, 0.5, 0.5, 0.6)
                ax2.bar(hist["bins"], hist["freq"], width=hist["width"],
                        color=grey_color, edgecolor=grey_color)
            line = content["line"]
            ax.plot(line["x"], line["y"], label=lab)
            if content['fitted_function'] is not None:
                fname = str(content['fitted_function']['function'])
                if fname[-2:] == '()':
                    fname = fname[:-2]
                fit_lab = f"Adjusted {fname}"
                if score_dict is not None:
                    fit_lab += f" ({score_dict['fitted']})"
                ax.plot(line["x"], content['fitted_function']["y"], '--',
                        label=fit_lab)
            ax.legend()
        if save:
            plt.savefig(f"{args.game}_{args.act_f}_{args.seed}.svg",
                        format="svg")
        else:
            plt.show()

    def best_fits(self, functions_list):
        pass

    def use_fitted_act(self):
        a1, b1, c1, d1 = self.act_func1.best_fitted_function_params
        a2, b2, c2, d2 = self.act_func2.best_fitted_function_params
        a3, b3, c3, d3 = self.act_func3.best_fitted_function_params
        a4, b4, c4, d4 = self.act_func4.best_fitted_function_params
        self.old_act1 = self.act_func1.activation_function
        self.old_act2 = self.act_func2.activation_function
        self.old_act3 = self.act_func3.activation_function
        self.old_act4 = self.act_func4.activation_function
        self.act_func1.activation_function = lambda x, u1, u2, u3: a1 * self.act_func1.best_fitted_function(c1 * torch.tensor(x) + d1) + b1
        self.act_func2.activation_function = lambda x, u1, u2, u3: a2 * self.act_func1.best_fitted_function(c2 * torch.tensor(x) + d2) + b2
        self.act_func3.activation_function = lambda x, u1, u2, u3: a3 * self.act_func1.best_fitted_function(c3 * torch.tensor(x) + d3) + b3
        self.act_func4.activation_function = lambda x, u1, u2, u3: a4 * self.act_func1.best_fitted_function(c4 * torch.tensor(x) + d4) + b4

    def restore_rational(self):
        self.act_func1.activation_function = self.old_act1
        self.act_func2.activation_function = self.old_act2
        self.act_func3.activation_function = self.old_act3
        self.act_func4.activation_function = self.old_act4

# if __name__ == '__main__':
#     model = Network([10], [10], "recrat")
#     model.to(torch.device("cuda:0"))
