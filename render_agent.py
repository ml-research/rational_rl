from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.environments import Atari
from mushroom_rl.utils.parameters import Parameter
import time
from utils import GymRenderer, make_deterministic, extract_game_name
from parsers import rendering_parser as parser
from collections import namedtuple
import json


def run_exp(agent, env, args):
    if args.no_display:
        renderer = None
    else:
        if args.record and args.video_title is None:
            args.video_title = args.agent_path.split("/")[-1].replace(".zip", "")
        renderer = GymRenderer(env, record=args.record, title=args.video_title)
    epsilon_test = Parameter(value=0.05)
    agent.policy.set_epsilon(epsilon_test)

    for i in range(1): # only 1 life
        total_r = 0
        state = env.reset()
        n_steps = 0
        while True:
            action = agent.draw_action(state)
            state, reward, done, _ = env.step(action)
            total_r += reward
            n_steps += 1
            if renderer is not None:
                renderer.render()
                time.sleep(0.01)
            if done:
                print("Done")
                break
        print("Total reward: " + str(total_r))
    if renderer is not None:
        renderer.close_recorder()


if __name__ == '__main__':
    args = parser.parse_args()

    game_name = extract_game_name(args.agent_path)
    with open(f'configs/{game_name}_config.json', 'r') as f:
        data = f'{json.load(f)}'.replace("'", '"')
        config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                history_length=config.history_length, max_no_op_actions=30)
    make_deterministic(args.seed, env)


    # agent_f = f"{args.algo}_{args.act_f}_{args.game}_s{args.seed}_e{args.epoch}.zip"
    print(f"Using agent from {args.agent_path}")
    agent = Agent.load(args.agent_path)

    run_exp(agent, env, args)
