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
args.epoch = 500
with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))


env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
            history_length=config.history_length, max_no_op_actions=30)
save_folder = 'images/af_comparisons/'
pau_type = args.act_f
limit = 20
if args.act_f == "rat":
    fig, axs = plt.subplots(1, 4, figsize=(12, 1.5))
    fig.tight_layout(pad=0.15)
else:
    fig, axs = plt.gcf(), plt.gca()
    fig.set_size_inches(3.2, 1.5)
for seed in [args.seed]:
    agents_folder = f'updated_populated_agents'
    filename = f"DQN_{args.act_f}_{args.game}_s{seed}_e{args.epoch}.zip"
    path = f"{agents_folder}/{filename}"
    print(path)
    ag = Agent.load(path)
    net = ag.approximator.model.network
    acts = [eval(f'net.act_func{act_n+1}') for act_n in range(4)]
    input_dists = [act.distribution for act in acts]
    if args.act_f == "rat":
        limit = compare_acts(acts, input_dists, str(seed), limit, axs, silu_comp=False)
    else:
        limit = compare_acts(acts[:1], input_dists, str(seed), limit, silu_comp=False)

plt.show()
out_filename = f"{args.game}_{pau_type}_seed_{args.seed}.svg"
fig.savefig(f"{save_folder + out_filename}", format='svg')
print(f"saved in {save_folder + out_filename}")
plt.close("all")
