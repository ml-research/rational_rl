import matplotlib.pyplot as plt
import pickle
import numpy as np
from parsers import graph_parser
import seaborn as sns


sns.set_style("whitegrid")
args = graph_parser.parse_args()

fig = plt.figure(figsize=(6, 4))


save_folder = f"scores/dqn_{args.game.lower()}"
base_filename = f"_scores{args.game}Deterministic-v4_seed"
act_funcs = ["paus", "rpau", "lrelu", "silu", "onlysilu", "DDQN"]
act_funcs_complete_names = ["PAUs", "Recurrent pau", "Leaky ReLU", "SiLU+dSiLU", "SiLU", "DDQN"]
nb_seeds = 5
min_seed_used = 10

for act, act_name in zip(act_funcs, act_funcs_complete_names):
    means = []
    # for seed_n in range(nb_seeds):
    for seed_n in range(nb_seeds):
        try:
            if act == "DDQN":
                data_from_file = pickle.load(open(f"{save_folder}/DDQN{base_filename}{seed_n}_lrelu.pkl", "rb"))
            else:
                data_from_file = pickle.load(open(f"{save_folder}/DQN{base_filename}{seed_n}_{act}.pkl", "rb"))
        except FileNotFoundError:
            # print(f"File {base_filename}{seed_n}_{act}.pkl not found")
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

    if act_name == "DDQN":
        lab = "DDQN LReLU"
    else:
        lab = f"DQN {act_name}"
    plt.plot(mean, label=lab)
    if args.game == "Asterix":
        plt.legend(fancybox=True, framealpha=0.5, fontsize=11.5)
    plt.fill_between(range(len(mean)), mean - standard_dev, mean + standard_dev,
                     alpha=0.5)


file_title = f"{args.game}_scores"
plt.xlabel("epochs")
plt.ylabel("score")
plt.title(file_title, fontsize=14)
if args.store:
    fig = plt.gcf()
    save_folder = "images/scores_graphs"
    fig.savefig(f"{save_folder}/{file_title}.svg")
    print(f"Saved in {save_folder}/{file_title}.svg")
else:
    plt.show()
