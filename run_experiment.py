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
from utils import get_stats, print_epoch, recover, sepprint, \
                  make_deterministic, save_approximator, \
                  RTPT, remove_heavy, load_activation_function
from parsers import parser
import json
from collections import namedtuple


args = parser.parse_args()

# exec(f'from configs.{args.game.lower()}_config import *')

###
with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))


file_name = config.game_name
scores = list()

optimizer = dict()
optimizer['class'] = optim.Adam
optimizer['params'] = dict(lr=.00025)

loaded_af = None

make_deterministic(args.seed)
file_name += "_seed" + str(args.seed)
init_epoch = 0
agent_save_dir = 'agent_save'
file_name += f"_{args.act_f}"

if args.freeze_pau:
    file_name += "_freezed"
    sepprint("Freezing")
if args.load:
    loaded_af = load_activation_function(args.act_f, args.game, args.seed)
    file_name += "_pretrained"

sepprint(f"Using {args.act_f.upper()} !")

# MDP
mdp = Atari(config.game_name, config.width, config.height, ends_at_life=True,
            history_length=config.history_length, max_no_op_actions=30)
mdp.seed(args.seed)

# Policy
epsilon = LinearParameter(value=1.,
                          threshold_value=.1,
                          n=1000000)
epsilon_test = Parameter(value=.05)
epsilon_random = Parameter(value=1)
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
    freeze_pau = args.freeze_pau,
    loaded_act_f = loaded_af
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

if args.algo == "DQN":
    agent = DQN(mdp.info, pi, approximator,
                approximator_params=approximator_params,
                **algorithm_params)
elif args.algo == "DDQN":
    agent = DoubleDQN(mdp.info, pi, approximator,
                      approximator_params=approximator_params,
                      **algorithm_params)

model = agent.approximator.model.network

agent.approximator.model.grad_norm_save = True

if args.recover:
    prefix = f'{args.algo}_{file_name}_epoch_n_'
    rec_filename, last_epoch = recover(prefix)
    if rec_filename is not None:
        #agent.load(f"{agent_save_dir}/{rec_filename}")
        agent = Agent.load(f"{agent_save_dir}/{rec_filename}")
        init_epoch = last_epoch + 1
        print("\n\n" + "+" * 30 + "\n\n")
        print("Recovered agent: " + rec_filename)
        print("\n\n" + "+" * 30 + "\n\n")
        with open(f'./{agent_save_dir}/scores_{file_name}_epoch_{last_epoch}.pkl',
                  'rb') as f:
            scores = pickle.load(f)
            print(f"found scores of size {len(scores)}")
    else:
        print("Cannot find any corresponding agent")

# Algorithm
core = Core(agent, mdp)

rtpt = RTPT(f"{config.game_name[:4]}S{args.seed}_{args.act_f}" , config.n_epochs)

# Fill replay memory with random dataset
if init_epoch == 0:
    rtpt.epoch_starts()
    print_epoch(0)
    #save_approximator(agent, f"./{agent_save_dir}/{args.algo}_{file_name}_epoch_n_-1")
    core.learn(n_steps=config.initial_replay_size,
               n_steps_per_fit=config.initial_replay_size)
    # Evaluate initial policy
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    dataset = core.evaluate(n_steps=config.test_samples, render=False)
    scores.append(get_stats(dataset))
    init_epoch += 1
    #save_approximator(agent, f"./{agent_save_dir}/{args.algo}_{file_name}_epoch_n_0")
    rtpt.setproctitle()

for n_epoch in range(init_epoch, config.n_epochs + 1):
    rtpt.epoch_starts()
    print_epoch(n_epoch)
    print('- Learning:')
    # learning step
    pi.set_epsilon(epsilon)
    mdp.set_episode_end(True)
    core.learn(n_steps=config.evaluation_frequency,
               n_steps_per_fit=config.train_frequency)
    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    dataset = core.evaluate(n_steps=config.test_samples)
    scores.append(get_stats(dataset))
    if n_epoch % 50 == 0:
        pi.set_epsilon(epsilon) # Important to save the current Linear Parameter in the agent
        agent.save(f"./{agent_save_dir}/{args.algo}_{file_name}_epoch_n_{n_epoch}")
        with open(f'./{agent_save_dir}/scores_{file_name}_epoch_{n_epoch}.pkl',
                  'wb') as f:
            pickle.dump(scores, f)
        #remove_heavy(f"./{agent_save_dir}/{args.algo}_{file_name}_epoch_n_{n_epoch-50}")
    #elif n_epoch in list(range(11)):
    #    save_approximator(agent, f"./{agent_save_dir}/{args.algo}_{file_name}_epoch_n_{n_epoch}")
    rtpt.setproctitle()


with open(f'./{agent_save_dir}/{args.algo}_scores{file_name}.pkl', 'wb') as f:
    pickle.dump(scores, f)
