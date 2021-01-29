import torch
import matplotlib.pyplot as plt
import pickle
import numpy as np
from activation_functions import SiLU, dSiLU
import sys

def plottable_bins(dist, thres):
    dataframe = dist.to_dataframe()
    dataframe = dataframe[dataframe["frequency"]/dataframe["frequency"].sum() > thres]
    bins = dataframe["left"].to_numpy()
    hist = dataframe["frequency"].to_numpy()
    return bins, hist


def visualize_act(act, epoch="", limit=1, args=None):
    # visualize activativation function
    range = (-limit, limit)
    xs = torch.linspace(*range, 100, device='cuda')
    ys = act(xs)
    plt.plot(xs.cpu().detach().numpy(),
             ys.cpu().detach().numpy(), 'b-')
    plt.xlabel("x")
    plt.ylabel("PAU(x)")
    plt.title(f"Shared PAU evolution through DQN learning on {args.game}")
    plt.legend(["Shared PAU: epoch n°"+epoch])
    return limit

def grid_search_minimal(f1, f2, bins, hist, scalea, scaleb, shifta, shiftb):
    min_comb = [scalea, scaleb, shifta, shiftb]
    min_dist = np.inf
    step = 0.1
    half_amount = 7
    lower_v = [val - half_amount * step for val in min_comb]
    upper_v = [val + half_amount * step for val in min_comb]
    for sca in np.arange(lower_v[0], upper_v[0], step):
        for scb in np.arange(lower_v[1], upper_v[1], step):
            for sha in np.arange(lower_v[2], upper_v[2], step):
                for shb in np.arange(lower_v[3], upper_v[3], step):
                    modified_func = lambda x: scb * f2(sca * x + shb) + sha
                    dist = neural_distance(f1, modified_func, bins, hist)
                    if dist < min_dist:
                        min_dist = dist
                        min_comb = [sca, scb, sha, shb]
                        # print(dist)
    return dist.item(), min_comb


def neural_distance(f1, f2, bins, hist):
    dist = 0
    for x, dens in zip(bins, hist):
        x = np.mean(x)
        if hasattr(f1, "denominator"):
            x1 = torch.tensor(x, device="cuda")
            y1 = f1(x1).item()
        else:
            print("NOT IMPLEMENTED 1")
        x2 = torch.tensor(x)
        y2 = f2(x2)
        dist += np.abs(y1 - y2)
    return dist/hist.sum()


def dist_all_acts(act, input_dists, scalea, scaleb, shifta, shiftb, activname=""):
    result = {}
    assert sys.argv[-2] in ["Asterix", "SpaceInvaders", "Tutankham"]
    print("dsilu")
    print("Seaching....")
    bins, hist = plottable_bins(sum(input_dists), 0.001)
    mini = grid_search_minimal(act, dSiLU(), bins, hist, scalea, scaleb, shifta, shiftb)
    print(mini)
    result["dsilu"] = mini
    print("ReLU")
    mini = grid_search_minimal(act, torch.nn.ReLU(), bins, hist, scalea, scaleb, shifta, shiftb)
    print(mini)
    result["ReLU"] = mini
    print("LeakyReLU")
    mini = grid_search_minimal(act, torch.nn.LeakyReLU(), bins, hist, scalea, scaleb, shifta, shiftb)
    print(mini)
    result["LeakyReLU"] = mini
    print("tanh")
    mini = grid_search_minimal(act, torch.nn.Tanh(), bins, hist, scalea, scaleb, shifta, shiftb)
    print(mini)
    result["tanh"] = mini
    print("sigmoid")
    mini = grid_search_minimal(act, torch.nn.Sigmoid(), bins, hist, scalea, scaleb, shifta, shiftb)
    print(mini)
    result["sigmoid"] = mini
    with open(f'Results_{sys.argv[-2]}{activname}.pkl', 'wb') as config_dictionary_file:
        pickle.dump(result, config_dictionary_file)
    exit()


def compare_acts(acts, input_dists, seed="", limit=1, axs=None, silu_comp=False):
    # visualize activativation function
    edc = (0.5, 0.5, 0.5, 0.6)
    if len(acts) > 1:
        for i, (ax, act, inp_dist) in enumerate(zip(axs.flat, acts, input_dists)):
            inp_dist = act.show(display=False)
            hist = inp_dist['hist']['freq']
            bins = inp_dist['hist']['bins']
            xs = inp_dist['line']['x']
            padding = (xs[-1] - xs[0])/2
            xs = torch.arange(xs[0] - padding, xs[-1] + padding,
                              xs[1] - xs[0]).cuda()
            ys = act(xs)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
            # if silu_comp:
            #     scale, shift = align_curves(act, dsilu, init_scale)
            #     ys2 = scaleb * dsilu(scalea * xs + shiftb) + shifta
            #     ax.plot(xs.cpu().detach().numpy(),
            #     ys2.cpu().detach().numpy(), 'r--', alpha=0.9,
            #     label=f"adjusted dSiLU")
            ax2 = ax.twinx()
            ax2.set_yticks([])
            ax2.bar(bins, hist, width=bins[1] - bins[0],
                    color=edc, edgecolor=edc)
            ax.plot(xs.cpu().detach().numpy(), ys.cpu().detach().numpy(), 'b-',
                    label=f"PAU{i+1}", alpha=0.7)
            # ax.set(xlabel='x', ylabel=f'PAU{i+1}(x)')
            # ax.legend(prop={'size': 13}, framealpha=0.5)
            ax.tick_params("both")
            # ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    else:
        act = acts[0]  # In shared_pau case, any act from acts is the same
        inp_dist = act.show(display=False)
        hist = inp_dist['hist']['freq']
        bins = inp_dist['hist']['bins']
        xs = inp_dist['line']['x']
        padding = (xs[-1] - xs[0])/2
        xs = torch.arange(xs[0] - padding, xs[-1] + padding,
                          xs[1] - xs[0]).cuda()
        ys = act(xs)
        ax = plt.gca()
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        # if silu_comp: # MODIFIED FOR DSILU
        #     init_scale = ys.max() - ys.min()
        #     scale, shift = align_curves(act, dsilu, init_scale)
        #     y2 = scaleb * dsilu(scalea * xs + shiftb) + shifta
        #     plt.plot(xs.cpu().detach().numpy(),
        #              y2.cpu().detach().numpy(), 'r--',
        #              label=f"adjusted dSiLU")
        ax.plot(xs.cpu().detach().numpy(), ys.cpu().detach().numpy(), 'b-',
                label=f"R.PAU", alpha=0.7)
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.bar(bins, hist, width=bins[1] - bins[0],
                color=edc, edgecolor=edc)
        # ax.legend(prop={'size': 13})
        ax.tick_params("both", labelsize="large")
    return limit


def visualize_multiple_act(acts, epoch="", limit=1, axs=None):
    # visualize activativation function
    range = (-limit, limit)
    xs = torch.linspace(*range, 100, device='cuda')
    for i, (ax, act) in enumerate(zip(axs.flat, acts)):
        ys = act(xs)
        ys2 = torch.sigmoid(xs)
        ax.plot(xs.cpu().detach().numpy(),
                ys.cpu().detach().numpy(), 'b-')
        ax.plot(xs.cpu().detach().numpy(),
                ys2.cpu().detach().numpy(), 'g-', alpha=0.4)
        ax.set(xlabel='x', ylabel=f'PAU{i+1}(x)')
    ylim = min([ax.get_ylim() for ax in axs.flat])
    for ax in axs.flat:
        # ax.set_ylim(ylim)
        ax.set_ylim([-1, 1])
    plt.legend(["PAU: epoch n°"+epoch])
    # plt.show()
    # exit()
    return limit


def visualize_multiple_heated_act(acts, input_dists, epoch="", limit=1, axs=None):
    # visualize activativation function
    for i, (ax, act, inp_dist) in enumerate(zip(axs.flat, acts, input_dists)):
        range = (-5, 5)
        # if inp_dist.total > 0:
        #     range = (inp_dist.min_edge, inp_dist.max_edge)
        xs = torch.linspace(*range, 1000, device='cuda')
        ys = act(xs)
        ys2 = torch.sigmoid(xs)
        ax.plot(xs.cpu().detach().numpy(),
                ys.cpu().detach().numpy(), 'b-')
        # ax.plot(xs.cpu().detach().numpy(),
        #         ys2.cpu().detach().numpy(), 'g-', alpha=0.4)
        if inp_dist.total > 0:
            bins = inp_dist.to_dataframe()["left"].to_numpy()
            hist = inp_dist.to_dataframe()["frequency"].to_numpy()
            hist = hist / hist.max()
            ax.bar(bins, hist, width = inp_dist.bin_sizes[0], color='grey', alpha=0.4)
        ax.set(xlabel='x', ylabel=f'PAU{i+1}(x)')
    # ylim = min([ax.get_ylim() for ax in axs.flat])
    for ax in axs.flat:
        # ax.set_ylim(ylim)
        ax.set_ylim([-0.3, 1])
        ax.set_xlim(range)
    plt.legend(["PAU: epoch n°"+epoch])
    # plt.show()
    # exit()
    return limit


def make_colored_line(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)
