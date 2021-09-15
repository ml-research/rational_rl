import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from parsers import graph_parser
import seaborn as sns

sns.set_style("whitegrid")
args = graph_parser.parse_args()

# fig = plt.figure(figsize=(3.7, 2))
fig = plt.figure(figsize=(16, 14))
SMOOTH = True

save_folder = f"scores/dqn_{args.game.lower()}"
base_filename = f"_scores{args.game}Deterministic-v4_seed"
# act_funcs = ["paus", "rpau", "lrelu", "d+silu", "onlysilu", "DDQN"]
# act_funcs_complete_names = ["RN", "RRN", "Leaky ReLU", "SiLU+dSiLU", "SiLU", "DDQN"]
act_funcs = ["paus", "rpau", "lrelu", "DDQN", "Rainbow_lrelu", "Rainbow_rat", "Rainbow_recrat"]
act_funcs_complete_names = ["RN", "RRN", "Leaky ReLU", "DDQN", "Rainbow LReLU", "Rainbow RN", "Rainbow RRN"]
# act_funcs = ["paus", "rpau", "lrelu", "DDQN"]
# act_funcs_complete_names = ["RN", "RRN", "LReLU", "LReLU"]
nb_seeds = 5
min_seed_used = 10
df_mlist, df_slist = [], []

for act, act_name in zip(act_funcs, act_funcs_complete_names):
    means = []
    for seed_n in range(nb_seeds):
        try:
            if act == "DDQN":
                data_from_file = pickle.load(open(f"{save_folder}/DDQN{base_filename}{seed_n}_lrelu.pkl", "rb"))
            elif "Rainbow" in act:
                act = act.split("_")[1]
                data_from_file = pickle.load(open(f"{save_folder}/Rainbow{base_filename}{seed_n}_{act}_300.pkl", "rb"))
            else:
                data_from_file = pickle.load(open(f"{save_folder}/DQN{base_filename}{seed_n}_{act}.pkl", "rb"))
        except FileNotFoundError:
            print(f"File {base_filename}{seed_n}_{act}.pkl not found")
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
    if act_name == "DDQN":
        lab = "DDQN LReLU"
    elif "Rainbow" in act_name:
        lab = act_name
        sty = "--"
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
    size = min(len(mean), 500)
    plt.plot(mean[:size], sty, label=lab)
    # if args.game == "Asterix":
    plt.legend(fancybox=True, framealpha=0.5, fontsize=11.5)
    if "Rainbow" not in act_name:
        plt.fill_between(range(size), mean[:size] - standard_dev[:size], mean[:size] + standard_dev[:size],
                         alpha=0.05)

if args.csv:
    complete_dfm = pd.concat(df_mlist, 1)
    complete_dfs = pd.concat(df_slist, 1)
    complete_dfm.to_csv(f"scores/csv/{args.game}_mean_scores.csv")
    complete_dfs.to_csv(f"scores/csv/{args.game}_std_scores.csv")
    exit()
file_title = f"{args.game}"
plt.xlabel("epochs")
plt.ylabel("score")
suffix = ["(min)", "(max)", "(mean)"][eval_metric_idx]
plt.title(f"{file_title} {suffix}", fontsize=14)
if args.store:
    fig = plt.gcf()
    save_folder = "images/rainbow_compare"
    fig.savefig(f"{save_folder}/{file_title}_scores.svg")
    print(f"Saved in {save_folder}/{file_title}_scores.svg")
else:
    plt.show()
