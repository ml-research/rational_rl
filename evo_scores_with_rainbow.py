import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import os
import sys

sns.set_style("whitegrid")


# fig = plt.figure(figsize=(3.7, 2))
fig, axes = plt.subplots(2, 4, figsize=(12, 4.5))
SMOOTH = True

lines_colour_cycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

games = ["Breakout", "Enduro", "Jamesbond", "Kangaroo", "Qbert", "TimePilot", "Seaquest", "SpaceInvaders"]
for game, ax in zip(games, axes.flatten()):
    save_folder = f"scores/dqn_{game.lower()}"
    base_filename = f"_scores{game}Deterministic-v4_seed"
    act_funcs = ["paus", "rpau", "lrelu", "DDQN", "Rainbow_rat", "Rainbow_recrat", "Rainbow_lrelu"]
    act_funcs_complete_names = ["RN", "RRN", "Leaky ReLU", "DDQN", "Rainbow RN", "Rainbow RRN", "Rainbow Leaky ReLU"]
    nb_seeds = 5
    min_seed_used = 10
    df_mlist, df_slist = [], []
    ax.axvline(200, ls=(0, (5, 5)), color="darkred", alpha=0.5)
    current_episodes = [450, 400, 350]
    for i, (act, act_name) in enumerate(zip(act_funcs, act_funcs_complete_names)):
        means = []
        for seed_n in range(nb_seeds):
            data_from_file = []
            if act == "DDQN":
                filename = f"{save_folder}/DDQN{base_filename}{seed_n}_lrelu.pkl"
            elif "Rainbow" in act:
                act_f = act.split("_")[1]
                filename = f"{save_folder}/Rainbow{base_filename}{seed_n}_{act_f}.pkl"
            else:
                filename = f"{save_folder}/DQN{base_filename}{seed_n}_{act}.pkl"
            if os.path.isfile(filename):
                data_from_file = pickle.load(open(filename, "rb"))
            elif "Rainbow" in act and seed_n == 0:
                for current_episode in current_episodes:
                    filename = f"{save_folder}/Rainbow_scores{game}Deterministic-v4_seed0_{act_f}_{current_episode}.pkl"
                    if os.path.isfile(filename):
                        print(f"FOUND UNFINISHED {filename}")
                        data_from_file = pickle.load(open(filename, "rb"))
                    else:
                        continue
            else:
                # print(f"File {base_filename}{seed_n}_{act}.pkl not found")
                continue
            # case where stats and all_scores are stored
            eval_metric_idx = 2  # min -> 0 ; max -> 1 ; mean -> 2
            if len(data_from_file[0]) == 2:
                means.append([scores[0][eval_metric_idx] for scores in data_from_file])
            # case where we stored all scores and no stats
            elif len(np.array(data_from_file).shape) == 1 and len(data_from_file[0]) > 10:
                means.append([np.mean(l) for l in data_from_file])
            # case where only stats are stored
            elif np.array(data_from_file).shape[1] == 4:
                means.append(np.array(data_from_file)[:, eval_metric_idx])
            else:
                print("Something went wrong, please check score storage")
                exit(1)
        if len(means) == 0:
            continue
        mini = min([len(sub) for sub in means])
        for i, subl in enumerate(means):
            means[i] = subl[:mini]

        if len(means) == 0:
            continue
        else:
            min_seed_used = min(min_seed_used, len(means))
        means = np.array(means)
        mean = means.mean(0)
        standard_dev = means.std(0)

        if SMOOTH: # Smooth
            coef = .9

            def smooth(scalars, weight):  # Weight between 0 and 1
                last = scalars[0]  # First value in the plot (first timestep)
                smoothed = list()
                for point in scalars:
                    smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
                    smoothed.append(smoothed_val)                        # Save it
                    last = smoothed_val                                  # Anchor the last smoothed value
                return np.array(smoothed)

            mean = smooth(mean, coef)
            standard_dev = smooth(standard_dev, coef)

        sty = "-"
        alpha = 1
        if act_name == "DDQN":
            lab = "Rigid DDQN"
            col = lines_colour_cycle[3]
            sty = "dotted"
        elif "Rainbow" in act_name:
            sty = "--"
            if "RRN" in act_name:
                col = lines_colour_cycle[1]
                lab = f"Rainbow (Reg. Plasticity)"
            elif "RN" in act_name:
                col = lines_colour_cycle[0]
                lab = f"Rainbow (Full Plasticity)"
            else:
                col = lines_colour_cycle[2]
                lab = f"Rigid Rainbow"
        else:
            if "RRN" in act_name:
                lab = f"DQN (Reg. Plasticity)"
                col = lines_colour_cycle[1]
            elif "RN" in act_name:
                lab = f"DQN (Full Plasticity)"
                col = lines_colour_cycle[0]
            else:
                lab = f"Rigid DQN"
                col = lines_colour_cycle[2]
                alpha = 0.8
        if "-csv" in sys.argv:
            df_mlist.append(pd.DataFrame({f'{act_name}': mean}))
            df_slist.append(pd.DataFrame({f'{act_name}': standard_dev}))
            continue
        size = min(len(mean), 500)
        ax.plot(mean[:size], linestyle=sty, color=col, label=lab, alpha=alpha)
        ax.fill_between(range(len(mean)), mean - standard_dev, mean + standard_dev,
        alpha=0.2)
        ax.set_title(game, fontsize=14)

handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, ncol=7, mode="expand", borderaxespad=0.,
           loc=(0.0, 0.0), fontsize=9.8, markerscale=0.8, handletextpad=0.3)

for ax in axes.flatten():
    yticks = ax.get_yticks()
    if ((yticks > 0) & (yticks < 1000)).any():
        pass
    elif ax.get_yticks().max() > 1000:
        ylabels = ['{:.0f}'.format(x) + 'K' for x in yticks/1000]
        ax.set_yticklabels(ylabels)

for i, ax in enumerate(axes[0]):
    ax.get_xaxis().set_ticklabels([])
file_title = f"rainbow_games"
fig.text(0.51, 0.085, 'epochs', ha='center', va='center', fontsize=12)
fig.text(0.01, 0.55, 'scores', ha='center', va='center', rotation='vertical',
         fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
suffix = ""
if "--store" in sys.argv:
    fig = plt.gcf()
    save_folder = "images"
    extension = ".pdf"
    fig.savefig(f"{save_folder}/{file_title}_scores{extension}")
    print(f"Saved in {save_folder}/{file_title}_scores{extension}")
else:
    plt.show()
