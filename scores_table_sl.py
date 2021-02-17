import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

rows = []
for dataset in ["cifar10"]:
    header = ["Network"]
    row = [dataset]
    for nn in ["vgg11", "vgg19"]: # vgg11
    # for nn in ["vgg11", "lenet"]: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        for act in ["lrelu", "rn", "rrn"]:
            all_scores = []
            for seed in range(5):
                filename = f"scores_{nn}_{act}_{seed}.pkl"
                sc_seed = pickle.load(open(f"{save_folder}/{filename}", "rb"))
                all_scores.append(sc_seed[1])
            header.append(f'{nn}_{act}')
            all_scores = np.array(all_scores)
            mean = np.mean(all_scores, 0)[-1]
            std = np.std(all_scores, 0)[-1]
            prec = 2
            row.append(f"{np.round(mean, prec)} ± {np.round(std, prec)}")
    rows.append(row)

score_df = pd.DataFrame(rows, columns=header)
print(score_df)
if "--store" in sys.argv:
    store_path = f"scores_tables/scores_table_supervized.csv"
    score_df.to_csv(store_path, index=False)
    print(f"stored in {store_path}")

rows = []
for dataset in ["imagenet"]:
    header = ["Network"]
    row = [dataset]
    for nn in ["mobilenet_v2", "resnet18"]: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        for act in ["lrelu", "rn", "rrn"]:
            all_scores = []
            for seed in range(5):
                filename = f"scores_{dataset}_{nn}_{act}_{seed}.pkl"
                try:
                    sc_seed = pickle.load(open(f"{save_folder}/{filename}", "rb"))
                except FileNotFoundError:
                    print(filename)
                    continue
                all_scores.append(sc_seed["train/accuracy@1"])
            header.append(f'{nn}_{act}')
            all_scores = np.array(all_scores)
            mean = np.mean(all_scores, 0)[-1]
            std = np.std(all_scores, 0)[-1]
            prec = 2
            row.append(f"{np.round(mean, prec)} ± {np.round(std, prec)}")
    rows.append(row)

score_df = pd.DataFrame(rows, columns=header)
print(score_df)
if "--store" in sys.argv:
    store_path = f"scores_tables/scores_table_supervized_imagenet.csv"
    score_df.to_csv(store_path, index=False)
    print(f"stored in {store_path}")
