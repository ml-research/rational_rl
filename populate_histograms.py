from utils import repair_agent
from mushroom_rl.algorithms.agent import Agent
import os
from mushroom_rl.environments import Atari
from rendering_atari import run_exp
import argparse
import json
from collections import namedtuple


def populate_histograms(agent, env, args):
    if hasattr(agent, "model"):
        network = agent.model.network
    elif hasattr(agent, "approximator"):
        network = agent.policy._approximator.model.network
    network.set_hists()
    args.record = False
    args.video_title = None
    args.no_display = True
    run_exp(agent, env, args)


def main():
    parser = argparse.ArgumentParser()

    agents_folder = f'agent_save'
    regex = f"DQN_*Deterministic-v4_SEED*_*_epoch_*"
    folder_list = [x[0] for x in os.walk(agents_folder)]
    # print(folder_list)
    # exit()
    for folder in folder_list:
        if not "DQN" in folder or "populated" in folder:
            continue
        ag = Agent.load(folder)
        args = parser.parse_args()
        # # for version compatibility
        game_info = folder.split("DQN_")[1]
        args.game = game_info.split("Deterministic")[0]
        args.seed = game_info.split("seed")[1][0]
        args.act_f = game_info.split(f"seed{args.seed}_")[1].split("_epoch_")[0]
        with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
            data = f'{json.load(f)}'.replace("'", '"')
            config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
        env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                    history_length=config.history_length, max_no_op_actions=30)
        repair_agent(ag)
        populate_histograms(ag, env, args)
        ag.save(folder)

if __name__ == '__main__':
    main()
