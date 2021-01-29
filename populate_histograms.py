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
import json
from collections import namedtuple
import os

saved_folder = "updated_agents"
populated_folder = "updated_populated_agents"
agents_list = [x for x in list(os.listdir(saved_folder)) if x not in list(os.listdir(populated_folder))]
agents_list.sort()

for agent_folder in agents_list:
    print(agent_folder)
    params = agent_folder.split("_")
    algo, act_f, game, seed, epoch = params
    game = game.split('Deterministic')[0]
    seed = int(seed[-1])
    epoch = epoch.split(".zip")[0][1:]
    if act_f not in ["rat", "recrat"]:
        continue
    with open(f'configs/{game.lower()}_config.json', 'r') as f:
        data = f'{json.load(f)}'.replace("'", '"')
        config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    file_name = config.game_name


    file_name += "_seed" + str(seed)

    epsilon_random = Parameter(value=1)
    epsilon_test = Parameter(value=.05)

    scores = list()

    optimizer = dict()
    optimizer['class'] = optim.Adam
    optimizer['params'] = dict(lr=.00025)

    sepprint(f"Using {act_f} !")

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
        activation_function=act_f,
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

    make_deterministic(seed, mdp)

    agent_f = f"{algo}_{act_f}_{game}_s{seed}_e{epoch}.zip"
    print(f"{saved_folder}/{agent_f}")
    agent = Agent.load(f"{saved_folder}/{agent_f}")

    net = agent.approximator.model.network
    net.input_retrieve_mode(max_saves=20000)

    core = Core(agent, mdp)
    print('- No Learning:')
    print('- Evaluation:')
    # evaluation step
    pi.set_epsilon(epsilon_test)
    mdp.set_episode_end(False)
    dataset = core.evaluate(n_steps=config.test_samples)
    scores = get_stats(dataset)
    agent.save(f"{populated_folder}/{agent_f}")
