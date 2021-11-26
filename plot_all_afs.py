import cupy as cp
from matplotlib import pyplot as plt
from visualize_net import compare_acts, visualize_multiple_act
from utils import list_files, repair_agent
import seaborn as sns
from mushroom_rl.algorithms.agent import Agent
# from populate_histograms import populate_histograms
import json
from collections import namedtuple
import os, sys


sns.set_style("whitegrid")

games = ["Asterix", "BattleZone", "Breakout", "Enduro", "Jamesbond", "Kangaroo",
         "Pong", "Qbert", "Seaquest", "Skiing", "SpaceInvaders", "Tennis",
         "TimePilot", "Tutankham", "VideoPinball"]

save_folder = 'images/af_comparisons/'
rat_type = "rat"
limit = 20
fig, axes = plt.subplots(len(games), 4, figsize=(10, 1.5*len(games)))
for ax in axes.flatten():
    # ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.tick_params(color=[1, 1, 1, 0])
agents_folder = f'updated_populated_agents'
for axs, game in zip(axes, games):
    for seed in range(5):
        filename = f"DQN_{rat_type}_{game}_s{seed}_e500.zip"
        path = f"{agents_folder}/{filename}"
        if os.path.isfile(path):
            print(f"Found agent at {path}")
            ag = Agent.load(path)
            break
    net = ag.approximator.model.network
    acts = [eval(f'net.act_func{act_n+1}') for act_n in range(4)]
    input_dists = [act.distribution for act in acts]
    axs[0].set_ylabel(game)
    if rat_type == "rat":
        from rational.torch import Rational
        Rational.list = acts
        for rat in acts:
            rat.snapshot_list = []
            rat.func_name = "  "
        Rational.show_all(axes=axs, display=False)

plt.tight_layout(pad=0.1)

if "--store" in sys.argv:
    out_filename = f"few_games_{rat_type}.svg"
    fig.savefig(f"{save_folder + out_filename}", format='svg')
    print(f"saved in {save_folder + out_filename}")
else:
    plt.show()
