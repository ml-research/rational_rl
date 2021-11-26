import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from parsers import graph_parser
import seaborn as sns

sns.set_style("whitegrid")
args = graph_parser.parse_args()

SMOOTH = True

games = ["Asterix", "BattleZone", "Breakout", "Enduro", "Jamesbond", "Kangaroo",
         "Pong", "Qbert", "Seaquest", "Skiing", "SpaceInvaders", "Tennis",
         "TimePilot", "Tutankham", "VideoPinball"]
act_funcs = ["paus", "rpau", "lrelu", "DDQN", "d+silu", "onlysilu"]
act_funcs_complete_names = ["RN", "RRN", "Leaky ReLU", "DDQN", "SiLU+dSiLU", "SiLU"]
# act_funcs = ["paus", "rpau", "lrelu", "3sp1p"]
# act_funcs_complete_names = ["RN", "RRN", "Leaky ReLU", "Mixed"]
# act_funcs = ["paus", "rpau", "lrelu", "DDQN"]
# act_funcs_complete_names = ["RN", "RRN", "LReLU", "LReLU"]
nb_seeds = 5
min_seed_used = 10
df_mlist, df_slist = [], []
fig, axes = plt.subplots(5, 3, figsize=(14, 17))
for game, ax in zip(games, axes.flatten()):
    save_folder = f"scores/dqn_{game.lower()}"
    base_filename = f"_scores{game}Deterministic-v4_seed"
    for act, act_name in zip(act_funcs, act_funcs_complete_names):
        means = []
        for seed_n in range(nb_seeds):
            try:
                if act == "DDQN":
                    data_from_file = pickle.load(open(f"{save_folder}/DDQN{base_filename}{seed_n}_lrelu.pkl", "rb"))
                else:
                    data_from_file = pickle.load(open(f"{save_folder}/DQN{base_filename}{seed_n}_{act}.pkl", "rb"))
            except FileNotFoundError:
                continue
            # case where stats and all_scores are stored
            if len(data_from_file[0]) == 2:
                means.append([scores[0][2] for scores in data_from_file])
            # case where we stored all scores and no stats
            elif len(np.array(data_from_file).shape) == 1 and len(data_from_file[0]) > 10:
                means.append([np.mean(l) for l in data_from_file])
            # case where only stats are stored
            elif np.array(data_from_file).shape[1] == 4:
                means.append(np.array(data_from_file)[:, 2])
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

        if act_name == "DDQN":
            lab = "DDQN LReLU"
        else:
            if "RN" in act_name:
                lab = f"DQN {act_name}"
            else:
                lab = None
                lab = f"DQN {act_name}"
        if args.csv:
            df_mlist.append(pd.DataFrame({f'{act_name}': mean}))
            df_slist.append(pd.DataFrame({f'{act_name}': standard_dev}))
            continue
        ax.plot(mean, label=lab)
        ax.fill_between(range(len(mean)), mean - standard_dev, mean + standard_dev,
        alpha=0.3)
        ax.set_title(game, fontsize=15)
        # if args.game == "Asterix":
axes[0][0].legend(fancybox=True, framealpha=0.5, fontsize=11.5)

fig.tight_layout(w_pad=-0.2)
for ax in axes.flatten():
    yticks = ax.get_yticks()
    if ((yticks > 0) & (yticks < 1000)).any():
        pass
    elif abs(ax.get_yticks()).max() > 1000:
        ylabels = ['{:.0f}'.format(x) + 'K' for x in yticks/1000]
        ax.set_yticklabels(ylabels)

if args.csv:
    complete_dfm = pd.concat(df_mlist, 1)
    complete_dfs = pd.concat(df_slist, 1)
    complete_dfm.to_csv(f"scores/csv/{args.game}_mean_scores.csv")
    complete_dfs.to_csv(f"scores/csv/{args.game}_std_scores.csv")
    exit()

# plt.title(file_title, fontsize=14)
file_title = "scores_evolutions"
if args.store:
    fig = plt.gcf()
    save_folder = "images/scores_graphs"
    fig.savefig(f"{save_folder}/{file_title}_all_games.pdf")
    print(f"Saved in {save_folder}/{file_title}_all_games.pdf")
else:
    plt.show()
