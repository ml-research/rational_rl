from matplotlib import pyplot as plt
from visualize_net import compare_acts, visualize_multiple_act
from utils import list_files, repair_agent
from parsers import parser
import seaborn as sns
from mushroom_rl.environments import Atari
from mushroom_rl.algorithms.agent import Agent
# from populate_histograms import populate_histograms
import json
from collections import namedtuple

sns.set_style("whitegrid")



args = parser.parse_args()
with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))


env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
            history_length=config.history_length, max_no_op_actions=30)
save_folder = 'images/af_comparisons/'
pau_type = args.act_f
limit = 20
if args.act_f == "paus":
    fig, axs = plt.subplots(1, 4, figsize=(12, 1.5))
    fig.tight_layout(pad=0.15)
else:
    fig, axs = plt.gcf(), plt.gca()
    fig.set_size_inches(6 , 4)
for seed in [args.seed]:
    agents_folder = f'agent_save/populated/'
    regex = f"DQN_{args.game}Deterministic-v4_seed{seed}_{ args.act_f}_epoch_*"
    files_list = list_files(regex, agents_folder)
    epochs = [int(ag_file.split("epoch_")[1]) for ag_file in files_list]
    epochs, files_list = zip(*sorted(zip(epochs, files_list)))
    assert len(epochs) > 0
    epochs, files_list = epochs[-1:], files_list[-1:]

    for i, filename in zip(epochs, files_list):
        path = f"{agents_folder}/{filename}"
        ag = Agent.load(path)
        # for version compatibility
        # repair_agent(ag)
        # populate_histograms(ag, env, args)
        input_dists = [eval(f'ag.policy._approximator.model.network.inp{act_n+1}') for act_n in range(4)]
        acts = [eval(f'ag.policy._approximator.model.network.act_func{act_n+1}') for act_n in range(4)]
        if args.act_f == "paus":
            limit = compare_acts(acts, input_dists, str(seed), limit, axs, silu_comp=False)
        else:
            limit = compare_acts(acts[:1], input_dists, str(seed), limit, silu_comp=False)

# plt.xlabel("x")
# plt.ylabel("PAU(x)")
# plt.suptitle(f"{args.act_f} on {args.game}")
# plt.legend()
plt.show()
out_filename = f"{args.game}_{pau_type}_seed_{args.seed}.svg"
fig.savefig(f"{save_folder + out_filename}", format='svg')
print(f"saved in {save_folder + out_filename}")
plt.close("all")
