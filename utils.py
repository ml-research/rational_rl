from mushroom_rl.utils.dataset import compute_metrics, compute_J
from os import listdir, makedirs, remove
from os.path import isfile, join
from rational.torch import Rational
import torch
import numpy as np
import re
import pickle
import datetime
from setproctitle import setproctitle
try:
    from gym.envs.classic_control import rendering
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
except:
    pass
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import copy


def make_deterministic(seed, mdp, states_dict=None):
    if states_dict is None:
        np.random.seed(seed)
        mdp.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Set all environment deterministic to seed {seed}")
    else:
        np.random.set_state(states_dict["numpy"])
        torch.random.set_rng_state(states_dict["torch"])
        mdp.env.env.np_random.set_state(states_dict["env"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Reset environment to recovered random state ")

def load_activation_function(act_f, game_name, seed):
    """
    load a pretrain act_f and freeze the weights.
    """
    save_dir = "agent_save/trained_functions"
    print(f"Loading {act_f} of {game_name} with seed {seed}")
    file_name = f"trained_{act_f}_{game_name}_{seed}.pkl"
    with open(f"{save_dir}/{file_name}", "rb") as store_file:
        act_fs = [pau.requires_grad_(False) for pau in pickle.load(store_file)]
    return act_fs


def recover(regex):
    filtered = list_files(f'checkpoint_{regex}', "checkpoints")
    try:
        last_epoch = max([int(ep.split("_epoch_")[-1]) for ep in filtered])
    except ValueError:
        print("ERROR: Could not find any corresponding file")
        exit(1)
    r2 = re.compile(".*" + str(last_epoch) + "$")
    filename = list(filter(r2.match, filtered))[0]
    sep = "+" * 50 + "\n"
    print(f"{sep}Found {filename} for recovery\n{sep}")
    agent, mdp, scores, states_dict = pickle.load(open(f"checkpoints/{filename}", 'rb'))
    return agent, mdp, scores, states_dict, last_epoch


def checkpoint(agent, mdp, scores, filename, epoch, agent_save_dir='agent_save'):
    """
    Creates a checkpoint with a tuple to be able to recover, remove the \
    last checkpoint and save approximator with few attributes
    """
    path = f"checkpoints/checkpoint_{filename}_epoch_{epoch}"
    makedirs("checkpoints", exist_ok=True)
    random_state_dict = {}
    random_state_dict["numpy"] = np.random.get_state()
    random_state_dict["torch"] = torch.random.get_rng_state()
    random_state_dict["env"] = mdp.env.env.np_random.get_state()
    pickle.dump((agent, mdp, scores, random_state_dict), open(path, 'wb'))
    if epoch > 50:
        remove(f"checkpoints/checkpoint_{filename}_epoch_{epoch-50}")
    save_agent(agent, f"{agent_save_dir}/{filename}_epoch_{epoch}")

def print_epoch(epoch):
    print('################################################################')
    print('Epoch: ', epoch)
    print('----------------------------------------------------------------')


def get_stats(dataset):
    score = compute_metrics(dataset)
    print(('min_reward: %f, max_reward: %f, mean_reward: %f, games_completed: %d' % score))
    return score


def list_files(regex, directory):
    r = re.compile(regex)
    all_saves = [f for f in listdir(directory)]
    filtered = list(filter(r.match, all_saves))
    return filtered



def sepprint(*args):
    print("\n" + "-" * 30)
    print(*args)
    print("\n" + "-" * 30)


def save_agent(agent, path):
    """
    Used in order to avoid using save method from mushroom_rl agent that saves
    way too much caracteristics
    """
    agent.save(path+".zip")


def remove_heavy(path):
    """
    Remove the memory and fit files from the folder
    """
    if path[-1] != "/":
        path += "/"
    try:
        remove(f"{path}_replay_memory.pickle")
        remove(f"{path}_fit.pickle")
    except FileNotFoundError:
        pass


def repair_agent(agent):
    if hasattr(agent, "model"):
        network = agent.model.network
    if hasattr(agent, "approximator"):
        network = agent.policy._approximator.model.network
    else:
        print("Not able to repair the agent, something went wrong")
        exit()
    if not hasattr(network.act_func1, "weight_numerator"):
        for i in range(4):
            exec(f"network.act_func{i+1}.weight_numerator = network.act_func{i+1}.numerator")
            exec(f"network.act_func{i+1}.weight_denominator = network.act_func{i+1}.denominator")
    else:
        for i in range(4):
            exec(f"network.act_func{i+1}.numerator = network.act_func{i+1}.weight_numerator")
            exec(f"network.act_func{i+1}.denominator = network.act_func{i+1}.weight_denominator")
    if not hasattr(network.act_func1, "center"):
        for i in range(4):
            exec(f"network.act_func{i+1}.center = 0")
            exec(f"network.act_func{i+1}.center = 0")


class GymRenderer():
    def __init__(self, env, record=False, title="video"):
        self.viewer = rendering.SimpleImageViewer()
        self.env = env
        self.record = record
        if record:
            self.video_rec = VideoRecorder(env.env, path=f"videos/{title}.mp4")

    def repeat_upsample(self, rgb_array, k=4, l=4, err=[]):
        # repeat kinda crashes if k/l are zero
        if rgb_array is None:
            raise ValueError("The rgb_array is None, probably mushroom_rl bug")
        if k <= 0 or l <= 0:
            if not err:
                print("Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l))
                err.append('logged')
            return rgb_array

        # repeat the pixels k times along the y axis and l times along the x axis
        # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

        return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

    def render(self, mode="zoomed"):
        if self.record:
            # self.env.render()
            self.video_rec.capture_frame()
        elif mode == "zoomed":
            rgb = self.env.render('rgb_array')
            upscaled = self.repeat_upsample(rgb, 4, 4)
            self.viewer.imshow(upscaled)
        else:
            self.env.render()

    def close_recorder(self):
        if self.record:
            self.video_rec.close()
            self.video_rec.enabled = False


class RTPT():
    """
    RemainingTimeToProcessTitle
    """
    def __init__(self, base_title, number_of_epochs, epoch_n=0):
        assert len(base_title) < 30
        self.base_title = "@" + base_title + "#"
        self._last_epoch_start = None
        self._epoch_n = epoch_n
        self._number_of_epochs = number_of_epochs
        setproctitle(self.base_title + "first_epoch")

    def epoch_starts(self):
        self._last_epoch_start = datetime.datetime.now()
        self._epoch_n += 1

    def setproctitle(self):
        last_epoch_duration = datetime.datetime.now() - self._last_epoch_start
        remaining_epochs = self._number_of_epochs - self._epoch_n
        remaining_time = str(last_epoch_duration * remaining_epochs).split(".")[0]
        if "day" in remaining_time:
            days = remaining_time.split(" day")[0]
            rest = remaining_time.split(", ")[1]
        else:
            days = 0
            rest = remaining_time
        complete_title = self.base_title + f"{days}d:{rest}"
        setproctitle(complete_title)


# def update_network(old_rational_net):
#     """
#     Will create a new rational_net from an old one.
#     """
#
#     new_net = model_cls(**kwargs)
#     import ipdb; ipdb.set_trace()
#     new_net.load_state_dict(copy.deepcopy(old_rational_net.state_dict()))
#     new_func1 = Rational()
#     new_func2 = Rational()
#     new_func3 = Rational()
#     new_func4 = Rational()
#     rational_net._h1.bias.requires_grad_(False)
#     rational_net._h2.bias.requires_grad_(False)
#     rational_net._h3.bias.requires_grad_(False)
#     rational_net._h4.bias.requires_grad_(False)
#     rational_net._h1.bias += rational_net.act_func1.center.item()
#     rational_net._h2.bias += rational_net.act_func2.center.item()
#     rational_net._h3.bias += rational_net.act_func3.center.item()
#     rational_net._h4.bias += rational_net.act_func4.center.item()
#     new_func1._from_old(rational_net.act_func1)
#     new_func2._from_old(rational_net.act_func2)
#     new_func3._from_old(rational_net.act_func3)
#     new_func4._from_old(rational_net.act_func4)
#     rational_net._h1.bias.requires_grad_(True)
#     rational_net._h2.bias.requires_grad_(True)
#     rational_net._h3.bias.requires_grad_(True)
#     rational_net._h4.bias.requires_grad_(True)
#     rational_net.act_func1 = new_func1
#     rational_net.act_func2 = new_func2
#     rational_net.act_func3 = new_func3
#     rational_net.act_func4 = new_func4
#
#     if not old_value == rational_net(inp).sum():
#         print("old value different from new one")
#         print(old_value)
#         print(rational_net(inp).sum())
#     rational_net.act_func1.input_retrieve_mode()
#     rational_net.act_func2.input_retrieve_mode()
#     rational_net.act_func3.input_retrieve_mode()
#     rational_net.act_func4.input_retrieve_mode()

# def update_pau_network(old_pau_net):
#     """
#     Will create a new rational_net from an old one.
#     """
#
#     new_net = model_cls(**kwargs)
#     import ipdb; ipdb.set_trace()
#     new_net.load_state_dict(copy.deepcopy(old_rational_net.state_dict()))
#     new_func1 = Rational()
#     new_func2 = Rational()
#     new_func3 = Rational()
#     new_func4 = Rational()
#     rational_net._h1.bias.requires_grad_(False)
#     rational_net._h2.bias.requires_grad_(False)
#     rational_net._h3.bias.requires_grad_(False)
#     rational_net._h4.bias.requires_grad_(False)
#     rational_net._h1.bias += rational_net.act_func1.center.item()
#     rational_net._h2.bias += rational_net.act_func2.center.item()
#     rational_net._h3.bias += rational_net.act_func3.center.item()
#     rational_net._h4.bias += rational_net.act_func4.center.item()
#     new_func1._from_old(rational_net.act_func1)
#     new_func2._from_old(rational_net.act_func2)
#     new_func3._from_old(rational_net.act_func3)
#     new_func4._from_old(rational_net.act_func4)
#     rational_net._h1.bias.requires_grad_(True)
#     rational_net._h2.bias.requires_grad_(True)
#     rational_net._h3.bias.requires_grad_(True)
#     rational_net._h4.bias.requires_grad_(True)
#     rational_net.act_func1 = new_func1
#     rational_net.act_func2 = new_func2
#     rational_net.act_func3 = new_func3
#     rational_net.act_func4 = new_func4
#
#     if not old_value == rational_net(inp).sum():
#         print("old value different from new one")
#         print(old_value)
#         print(rational_net(inp).sum())
#     rational_net.act_func1.input_retrieve_mode()
#     rational_net.act_func2.input_retrieve_mode()
#     rational_net.act_func3.input_retrieve_mode()
#     rational_net.act_func4.input_retrieve_mode()
