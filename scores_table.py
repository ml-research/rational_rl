import pandas as pd
import pickle
import os
import numpy as np
import sys


def format_float(number):
    size = len(str(abs(int(number))))
    if size > 2:
        return f"{number:.1f}"
    elif size == 2:
        return f"{number:.1f}"
    elif size == 1:
        return f"{number:.2f}"
    else:
        print("Inconsistent number")
        exit(1)

def format_float_std(number):
    size = len(str(abs(int(number))))
    if size > 2:
        return f"{number:.0f}"
    elif size == 2:
        return f"{number:.1f}"
    elif size == 1:
        return f"{number:.1f}"
    else:
        print("Inconsistent number")
        exit(1)

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
    act_fs = ["lrelu", "onlysilu", "d+silu", "paus", "rpau", "DDQN"]
    act_names = ["LReLU", "SiLU", "d+SiLU", "RN",  "RRN", "DDQN"]
    suffix = "all"

elif "--pretrained" in sys.argv:
    act_fs = ["lrelu", "paus", "paus_pretrained", "rpau"]
    act_names = ["lrelu", "flexible", "pretrained", "recurrent"]
    suffix = "pretrained"
else:
    print("Please provide the activation functions for the table:")
    print("    python3 scores_table.py --pretrained (--store)" )
    print("or  python3 scores_table.py --all (--store)")
    exit(1)

if "--random" in sys.argv:
    act_fs.append("Random")
    act_names.append("Random")
    suffix += "_with_random"

games_dirs = sorted(os.listdir("scores"))
means = []
stds = []
for game_dir in games_dirs:
    if "ddqn" in game_dir:  # exclude beam rider that is empty
        continue
    game_name = game_dir.split("_")[1]
    scores_on_game = []
    std_on_game = []
    std_on_game.append(game_name.capitalize())
    scores_on_game.append(game_name.capitalize())
    for act in act_fs:
        avg = []
        nb_seeds = 0
        for score_file in sorted(os.listdir(f"scores/{game_dir}")):
            if (f"{act}.pkl" in score_file and score_file[:3] == "DQN") or score_file[:4] == act:
                with open(f"scores/{game_dir}/{score_file}", "rb") as f:
                    data_from_file = pickle.load(f)
                if "--max" in sys.argv:
                    if len(data_from_file[0]) == 2:
                        agent_means = [scores[0][1] for scores in data_from_file]
                    # case where we stored all scores and no stats
                    elif len(np.array(data_from_file).shape) == 1 and len(data_from_file[0]) > 10:
                        agent_means = [np.max(l) for l in data_from_file]
                    # case where only stats are stored
                    elif np.array(data_from_file).shape[1] == 4:
                        agent_means = np.array(data_from_file)[:, 1]
                    else:
                        print("Problem with the score storage.")
                        exit(1)
                    avg.append(np.max(agent_means))
                    nb_seeds += 1
                else:
                    if len(data_from_file[0]) == 2:
                        agent_means = [scores[0][2] for scores in data_from_file]
                    # case where we stored all scores and no stats
                    elif len(np.array(data_from_file).shape) == 1 and len(data_from_file[0]) > 10:
                        agent_means = [np.mean(l) for l in data_from_file]
                    # case where only stats are stored
                    elif np.array(data_from_file).shape[1] == 4:
                        agent_means = np.array(data_from_file)[:, 2]
                    else:
                        print("Problem with the score storage.")
                        exit(1)
                    avg.append(agent_means[-1])
                    nb_seeds += 1
            elif score_file[:6] == act == "Random":  # specifically for random
                with open(f"scores/{game_dir}/{score_file}", "rb") as f:
                    data_from_file = pickle.load(f)
                    avg.append(data_from_file[2])
                    nb_seeds += 1
            if "DQN" in score_file:
                game_name = score_file.split("scores")[1].split("Deter")[0]

        if nb_seeds > 0:
            if "--max" in sys.argv:
                if np.abs(np.mean(avg)) < 10:
                    as_max = int(np.max(avg))  # all seeds mean
                elif np.abs(np.mean(avg)) < 100:
                    as_max = int(np.max(avg))  # all seeds mean
                else:
                    as_max = int(np.max(avg))  # all seeds mean
                scores_on_game.append(f"{as_max}")
            else:
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
                    rand_std = np.std(rand_score)
                    rand_score = np.average(rand_score)
                    human_score = human_scores[game_name.lower()]
                    normalize_score = format_float(100 * (as_mean - rand_score) / (human_score - rand_score))
                    normalize_std = format_float_std(100 * (as_std - rand_std) / (human_score - rand_score))
                    scores_on_game.append(f"{normalize_score} ± {normalize_std}")
                    # scores_on_game.append(normalize_score))
                    std_on_game.append(normalize_std)
                else:
                    scores_on_game.append(f"{as_mean} ± {as_std}")
        else:
            scores_on_game.append(np.nan)
    means.append(scores_on_game)
    stds.append(std_on_game)

cols = ["Game"] + act_names
if "--seed" in sys.argv:
    cols += ["# seeds"]
score_df = pd.DataFrame(means, columns=cols).dropna()
# if "--pretrained" in sys.argv:
#     score_df = score_df.T
if "--human_normalized" in sys.argv:
    suffix += "_human_normalized"
print(score_df)
if "--store" in sys.argv:
    if "--max" in sys.argv:
        score_df.to_csv(f"scores_tables/scores_table_max_{suffix}.csv", index=False)
        print(f"stored in scores_tables/scores_table_max_{suffix}.csv")
    else:
        score_df.to_csv(f"scores_tables/scores_table_{suffix}.csv", index=False)
        print(f"stored in scores_tables/scores_table_{suffix}.csv")
        if "--human_normalized" in sys.argv:
            stds_df = pd.DataFrame(stds, columns=cols).dropna()
            stds_df.to_csv(f"scores_tables/stds_table_{suffix}.csv", index=False)
            print(f"stored in scores_tables/stds_table_{suffix}.csv")
# print(score_df)
