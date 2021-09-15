import pandas as pd
import pickle
import os
import numpy as np
import sys

human_scores = {
    "asterix": 7536.00,
    "battlezone": 33030.00,
    "breakout": 27.90,
    "enduro": 740.20,
    "jamesbond": 368.50,
    "kangaroo": 2739.00,
    "pong": 15.50,
    "qbert": 12085.00,
    "seaquest": 40425.80,
    "skiing": -3686.60,
    "spaceinvaders": 1464.90,
    "tennis": -6.70,
    "timepilot": 5650.00,
    "tutankham": 138.30,
    "videopinball": 15641.10
}

if "--all" in sys.argv:
    act_fs = ["lrelu", "rpau", "paus", "onlysilu", "d+silu", "Random", "DDQN"]
    act_names = ["LReLU", "RRN", "RN",  "SiLU", "d+SiLU", "Random", "DDQN"]
    suffix = "all"
elif "--pretrained" in sys.argv:
    act_fs = ["paus", "paus_pretrained"]
    act_names = ["flexible", "pretrained"]
    suffix = "pretrained"
else:
    print("Please provide the activation functions for the table:")
    print("    python3 scores_table.py --pretrained (--store)" )
    print("or  python3 scores_table.py --all (--store)")
    exit(1)

games_dirs = sorted(os.listdir("scores"))
rows = []
for game_dir in games_dirs:
    if "ddqn" in game_dir:  # exclude beam rider that is empty
        continue
    game_name = game_dir.split("_")[1]
    scores_on_game = []
    scores_on_game.append(game_name)
    min_seed = 100
    for act in act_fs:
        avg = []
        nb_seeds = 0
        for score_file in sorted(os.listdir(f"scores/{game_dir}")):
            if (f"{act}.pkl" in score_file and score_file[:3] == "DQN") or score_file[:4] == act:
                with open(f"scores/{game_dir}/{score_file}", "rb") as f:
                    data_from_file = pickle.load(f)
                if len(data_from_file[0]) == 2:
                    means = [scores[0][2] for scores in data_from_file]
                # case where we stored all scores and no stats
                elif len(np.array(data_from_file).shape) == 1 and len(data_from_file[0]) > 10:
                    means = [np.mean(l) for l in data_from_file]
                # case where only stats are stored
                elif np.array(data_from_file).shape[1] == 4:
                    means = np.array(data_from_file)[:, 2]
                else:
                    print("Problem with the score storage.")
                    exit(1)
                avg.append(means[-1])
                nb_seeds += 1
            elif score_file[:6] == act == "Random":  # specifically for random
                with open(f"scores/{game_dir}/{score_file}", "rb") as f:
                    data_from_file = pickle.load(f)
                    avg.append(data_from_file[2])
                    nb_seeds += 1
            if "DQN" in score_file:
                game_name = score_file.split("scores")[1].split("Deter")[0]

        if nb_seeds > 0:
            if np.abs(np.mean(avg)) < 10:
                as_mean = np.round(np.mean(avg), 2)  # all seeds mean
                as_std = np.round(np.std(avg), 2)  # all seeds mean
            elif np.abs(np.mean(avg)) < 100:
                as_mean = np.round(np.mean(avg), 1)  # all seeds mean
                as_std = np.round(np.std(avg), 1)  # all seeds mean
            else:
                as_mean = np.round(np.mean(avg), 0)  # all seeds mean
                as_std = np.round(np.std(avg), 0)  # all seeds mean
            if "--human_normalized" in sys.argv:
                rand_score = []
                for i in range(5):
                    score_file = f"Random_scores{game_name}Deterministic-v4_seed{i}.pkl"
                    with open(f"scores/{game_dir}/{score_file}", "rb") as f:
                        data_from_file = pickle.load(f)
                        rand_score.append(data_from_file[2])
                rand_score = np.average(rand_score)
                human_score = human_scores[game_name.lower()]
                scores_on_game.append(100 * (as_mean - rand_score) / (human_score - rand_score))
            else:
                scores_on_game.append(f"{as_mean} ± {as_std}")
        else:
            scores_on_game.append(np.nan)
            min_seed = 0
        min_seed = min(min_seed, nb_seeds)
    if "--seed" in sys.argv:
        scores_on_game.append(min_seed)
    rows.append(scores_on_game)

cols = ["game"] + act_names
if "--seed" in sys.argv:
    cols += ["# seeds"]
score_df = pd.DataFrame(rows, columns=cols).dropna()
# if "--pretrained" in sys.argv:
#     score_df = score_df.T
if "--human_normalized" in sys.argv:
    suffix += "_human_normalized"
print(score_df)
if "--store" in sys.argv:
    score_df.to_csv(f"scores_tables/scores_table_{suffix}.csv", index=False)
# print(score_df)
