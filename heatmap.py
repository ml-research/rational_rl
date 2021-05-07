from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.environments import Atari
from mushroom_rl.utils.parameters import Parameter
import time
from utils import GymRenderer, make_deterministic, extract_game_name
from parsers import gradcam_parser as parser
from collections import namedtuple
import json
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import glob
import matplotlib.pyplot as plt


all_agents = ["lrelu", "rat", "recrat"]


def load_agents(game_name, agents_types):
    agents = {}
    for path in glob.glob("updated_agents/*"):
        if game_name.lower() in path.lower():
            for agt in agents_types:
                if agt in path.lower() and agt not in agents.keys():
                    agents[agt] = Agent.load(path)
                    break
    return agents


def run_exp(agents, env, args):
    if args.record and args.video_title is None:
        args.video_title = args.agent_path.split("/")[-1].replace(".zip", "")
    renderer = GymRenderer(env, record=args.record, title=args.title)
    epsilon_test = Parameter(value=0.05)
    nets = {}
    cams = {}
    if len(agents) > 1:
        main_agent = agents["recrat"]
    else:
        main_agent = agents[[ag for ag in agents.keys()][0]]
    for agt in agents.keys():
        agent = agents[agt]
        agent.policy.set_epsilon(epsilon_test)
        net = agent.approximator.model.network
        nets[agt] = net
        cams[agt] = GradCAM(model=net, target_layer=net._h1, use_cuda=True)

    for i in range(1): # only 1 life
        total_r = 0
        state = env.reset()
        n_steps = 0
        while True:
            if n_steps < 100:
                n_steps += 1
                action = main_agent.draw_action(state)
                state, reward, done, _ = env.step(action)
                continue
            with torch.no_grad():
                action = main_agent.draw_action(state)
            # action = np.array([env.env.action_space.sample()])
            state, reward, done, _ = env.step(action)
            if n_steps % 10 == 0:
                input_tensor = torch.cat([torch.tensor(state._frames[0]).unsqueeze(0) for i in range(4)]).unsqueeze(0).cuda()
                fig, axes = plt.subplots(1, 3)
                for i, (agt, cam) in enumerate(cams.items()):
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        target_category=None) # the highest
                    heated_img = renderer.render("heatmap", grayscale_cam)
                    ax = axes[i]
                    ax.imshow(heated_img)
                    ax.set_title(agt)
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                plt.show()
            total_r += reward
            n_steps += 1
            # if renderer is not None:
            #     renderer.render("return")
            #     time.sleep(0.01)
            if done:
                print("Done")
                break
        print("Total reward: " + str(total_r))
    if renderer is not None:
        renderer.close_recorder()


if __name__ == '__main__':
    args = parser.parse_args()

    with open(f'configs/{args.game_name.lower()}_config.json', 'r') as f:
        data = f'{json.load(f)}'.replace("'", '"')
        config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                history_length=config.history_length, max_no_op_actions=30)
    make_deterministic(args.seed, env)
    if args.agent == "all":
        args.agent = all_agents
    agents = load_agents(args.game_name, args.agent)

    run_exp(agents, env, args)
