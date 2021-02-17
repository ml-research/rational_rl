import pickle
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.algorithms.value import DQN, DoubleDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import Atari
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from networks import Network, USE_CUDA
from utils import get_stats, sepprint, make_deterministic
from parsers import eval_parser as parser
import json
from collections import namedtuple


args = parser.parse_args()

assert args.random is not None or (args.algo is not None and args.act_f is not None)
if args.random:
    args.act_f = "lrelu"

with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))


file_name = config.game_name

file_name += "_seed" + str(args.seed)
agent_save_dir = 'updated_agents'


epsilon_random = Parameter(value=1)
epsilon_test = Parameter(value=.05)

scores = list()

optimizer = dict()
optimizer['class'] = optim.Adam
optimizer['params'] = dict(lr=.00025)

sepprint(f"Using {args.act_f.upper()} !")

# MDP
mdp = Atari(config.game_name, config.width, config.height, ends_at_life=True,
            history_length=config.history_length, max_no_op_actions=30)

# Policy
epsilon = LinearParameter(value=1.,
                          threshold_value=.1,
                          n=1000000)

pi = EpsGreedy(epsilon=epsilon_random)

# Approximator
input_shape = (config.history_length, config.height, config.width)
approximator_params = dict(
    network=Network,
    input_shape=input_shape,
    output_shape=(mdp.info.action_space.n,),
    n_actions=mdp.info.action_space.n,
    n_features=Network.n_features,
    optimizer=optimizer,
    loss=F.smooth_l1_loss,
    use_cuda=USE_CUDA,
    activation_function=args.act_f,
    freeze_pau = False,
    loaded_act_f = False
)

approximator = TorchApproximator

# Agent
algorithm_params = dict(
    batch_size=32,
    target_update_frequency=config.target_update_frequency // config.train_frequency,
    replay_memory=None,
    initial_replay_size=config.initial_replay_size,
    max_replay_size=config.max_replay_size
)

make_deterministic(args.seed, mdp)

if args.random:
    agent = DQN(mdp.info, pi, approximator,
                approximator_params=approximator_params,
                **algorithm_params)
else:
    agent_f = f"{args.algo}_{args.act_f}_{args.game}_s{args.seed}_e{args.epoch}.zip"
    print(f"{agent_save_dir}/{agent_f}")
    agent = Agent.load(f"{agent_save_dir}/{agent_f}")

net = agent.approximator.model.network
net.input_retrieve_mode()

core = Core(agent, mdp)
print('- No Learning:')
print('- Evaluation:')
# evaluation step
if args.random:
    pi.set_epsilon(epsilon_random)
else:
    pi.set_epsilon(epsilon_test)
mdp.set_episode_end(False)
dataset = core.evaluate(n_steps=config.test_samples)
scores = get_stats(dataset)
print(scores)
net.show()

# with open(f'./{agent_save_dir}/Random_scores{file_name}.pkl', 'wb') as f:
#     pickle.dump(scores, f)
