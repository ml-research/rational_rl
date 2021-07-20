import pickle
import torch.optim as optim
import torch.nn.functional as F
from mushroom_rl.algorithms.value import DQN, DoubleDQN, CategoricalDQN, Rainbow, DuelingDQN
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Core
from mushroom_rl.environments import Atari
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from mushroom_rl.utils.replay_memory import PrioritizedReplayMemory as PriorityReplay
from networks import Network, USE_CUDA
from utils import get_stats, recover, sepprint, print_epoch, \
                  make_deterministic, checkpoint, \
                  RTPT, load_activation_function
from parsers import parser
import json
from collections import namedtuple


args = parser.parse_args()

with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
    data = f'{json.load(f)}'.replace("'", '"')
    config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))


file_name = config.game_name

loaded_af = None
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



epsilon_test = Parameter(value=.05)
epsilon_random = Parameter(value=1)

if not args.recover:
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

    #set isFeatureNetwork to true if we have DuelingDQN or DistribDQN, because those need a different Network
    isFeatureNetwork = False
    if args.algo == "DistribDQN" or args.algo == "DuelingDQN":
        isFeatureNetwork = True

    approximator_params = dict(
        network=Network,
        input_shape=input_shape,
        output_shape=(mdp.info.action_space.n,),
        n_actions=mdp.info.action_space.n,
        n_features=Network.n_features,
        optimizer=optimizer,
        loss=F.smooth_l1_loss,
        use_cuda=USE_CUDA,
        #isFeatureNetwork=isFeatureNetwork,
        activation_function=args.act_f,
        freeze_pau = args.freeze_pau,
        loaded_act_f = loaded_af
    )

    approximator = TorchApproximator


    #check whether we use prioritized replay or not
    wantPrioritized = args.prio or args.algo == "Rainbow"
    if wantPrioritized:
        #parameters chosen like in rainbow paper
        #n maybe needs to be tweaked
        beta = LinearParameter(0.4, 1.0, n=1000000)
        replay_memory = PriorityReplay(config.initial_replay_size, config.max_replay_size, 0.5, beta)
    else:
        replay_memory = None

    # Agent
    algorithm_params = dict(
        batch_size=32,
        target_update_frequency=config.target_update_frequency // config.train_frequency,
        replay_memory=replay_memory,
        initial_replay_size=config.initial_replay_size,
        max_replay_size=config.max_replay_size
    )

    make_deterministic(args.seed, mdp)

    if args.algo == "DQN":
        agent = DQN(mdp.info, pi, approximator,
                    approximator_params=approximator_params,
                    **algorithm_params)
    elif args.algo == "DDQN":
        agent = DoubleDQN(mdp.info, pi, approximator,
                          approximator_params=approximator_params,
                          **algorithm_params)
    elif args.algo == "Rainbow":
        algorithm_params["initial_replay_size"] = 80000
        algorithm_params["beta"] = beta
        algorithm_params["n_steps_return"] = 3
        algorithm_params["alpha_coeff"] = 0.5
        agent = Rainbow(mdp.info, pi, approximator_params=approximator_params, n_atoms=51, v_min=-10, v_max=10, **algorithm_params)
    elif args.algo == "DistribDQN":
        agent = CategoricalDQN(mdp.info, pi, approximator_params=approximator_params, n_atoms=51, v_min=-10, v_max=10, **algorithm_params)
    elif args.algo == "DuelingDQN":
        agent = DuelingDQN(mdp.info, pi, approximator_params=approximator_params, **algorithm_params)

    model = agent.approximator.model.network

    agent.approximator.model.grad_norm_save = True

else:
    prefix = f'{args.algo}_{file_name}_epoch_'
    agent, mdp, scores, states_dict, last_epoch = recover(prefix)
    make_deterministic(None, mdp, states_dict)
    pi = agent.policy
    epsilon = pi._epsilon
    init_epoch = last_epoch + 1


core = Core(agent, mdp)

# # import ipdb; ipdb.set_trace()
# import torch; torch.set_printoptions(precision=15)
# rat1 = agent.approximator.model.network.act_func1
# print(rat1.numerator)
# def printgradnorm(self, grad_input, grad_output):
#     print('Inside ' + self.__class__.__name__ + ' backward')
#     print('Inside class:' + self.__class__.__name__)
#
# rat1.register_backward_hook(printgradnorm)

rtpt = RTPT(f"{config.game_name[:4]}S{args.seed}_{args.act_f}", config.n_epochs)
for epoch in range(init_epoch, config.n_epochs + 1):
    rtpt.epoch_starts()
    print_epoch(epoch)
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
    if epoch % 50 == 0:
        pi.set_epsilon(epsilon)
        checkpoint(agent, mdp, scores, f"{args.algo}_{file_name}", epoch)
        print("Saving the agent")
        with open(f'./{agent_save_dir}/{args.algo}_scores{file_name}_{epoch}.pkl', 'wb') as f:
            pickle.dump(scores, f)
    rtpt.setproctitle()


with open(f'./{agent_save_dir}/{args.algo}_scores{file_name}.pkl', 'wb') as f:
    pickle.dump(scores, f)
