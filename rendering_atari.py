from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.environments import Atari
from mushroom_rl.utils.parameters import Parameter
import time
from utils import repair_agent, GymRenderer, make_deterministic
from parsers import rendering_parser as parser
from collections import namedtuple
import json

def run_exp(agent, env, args):
    if args.no_display:
        renderer = None
    else:
        if args.record and args.video_title is None:
            args.video_title = f"{args.algo}_{args.game}_{args.act_f}"
        renderer = GymRenderer(env, record=args.record, title=args.video_title)
    if "pau" in args.act_f:
        repair_agent(agent)
    epsilon_test = Parameter(value=0.05)
    agent.policy.set_epsilon(epsilon_test)

    for i in range(3): # only 3 lives max
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

    with open(f'configs/{args.game.lower()}_config.json', 'r') as f:
        data = f'{json.load(f)}'.replace("'", '"')
        config = json.loads(data, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    env = Atari(config.game_name, config.width, config.height, ends_at_life=True,
                history_length=config.history_length, max_no_op_actions=30)
    env.seed(args.seed)
    make_deterministic(2)


    agent_f = f"{args.algo}_{args.game}Deterministic-v4_seed{args.seed}_{args.act_f}_epoch_{args.epoch}"
    agent = Agent.load(f"agent_save/{agent_f}")

    run_exp(agent, env, args)
