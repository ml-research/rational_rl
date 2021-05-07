import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-af", "--act", help="Activation function to use",
                    action="store", dest="act_f", required=True,
                    choices=['rat', 'recrat', 'lrelu', 'relu', 'silu',
                             'd+silu', 'r3r', 'r2r2', 'rr3', 'r2rr'])
parser.add_argument("-alg", "--algo", help="Activation function to use",
                    action="store", dest="algo", required=True,
                    choices=['DQN', 'DDQN', 'DistribDQN', 'DuelingDQN'])
parser.add_argument("-g", "--game", help="Game to train on", required=True,
                    action="store", dest="game")
parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                    required=True, action="store", dest="seed", type=int)
parser.add_argument("--freeze", help="Freeze pau that is then non learnable",
                    action="store_true", dest="freeze_pau", default=False)
parser.add_argument("--recover", help="Recover from the last trained agent",
                    action="store_true", dest="recover", default=False)
parser.add_argument("--load", help="Load a pretrained AF from a folder",
                    action="store_true", dest="load", default=False)
parser.add_argument("-prio", help="Whether to use Prioritized Replay used in the DeepMind Paper", dest="prio", action="store_true",
                    required=False, default=None)

graph_parser = argparse.ArgumentParser()
graph_parser.add_argument("-g", "--game", help="Game to train on",
                          required=True, action="store", dest="game")
graph_parser.add_argument("-s", "--store", help="Save the graph in svg file (instead of displaying)",
                          action="store_true", dest="store")
graph_parser.add_argument("--csv_score", help="Creates a file csv file of the result",
                          action="store_true", dest="csv")

rendering_parser = argparse.ArgumentParser()
rendering_parser.add_argument('agent_path', help='path to agent to vizualize')
rendering_parser.add_argument("-r", "--record", help="records the video",
                              action="store_true", dest="record", default=False)
rendering_parser.add_argument("-nd", "--no_display", help="Avoid displaying",
                              action="store_true", dest="no_display", default=False)
rendering_parser.add_argument("--video_title", help="Video title the video",
                              dest="video_title", default=None)
rendering_parser.add_argument("-e", "--epoch", help="Epoch to use",
                              dest="epoch", default="500")
rendering_parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                              default=0, action="store", dest="seed", type=int)

eval_parser = argparse.ArgumentParser()
eval_parser.add_argument("-af", "--act", help="Activation function to use",
                         action="store", dest="act_f",
                         choices=['rat', 'recrat', 'lrelu', 'relu', 'silu',
                                  'd+silu'])
eval_parser.add_argument("-alg", "--algo", help="Activation function to use",
                         action="store", dest="algo",
                         choices=['DQN', 'DDQN'])
eval_parser.add_argument("-g", "--game", help="Game to train on", required=True,
                         action="store", dest="game")
eval_parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                         required=True, action="store", dest="seed", type=int)
eval_parser.add_argument("--random", help="Evaluate random agent instead",
                         action="store_true", dest="random")
eval_parser.add_argument("--graph_save", help="Save the graph",
                         action="store_true", dest="save")
eval_parser.add_argument("-e", "--epoch", help="Epoch to use",
                         dest="epoch", default="500")

gradcam_parser = argparse.ArgumentParser()
gradcam_parser.add_argument('game_name', help='Name of the game')
gradcam_parser.add_argument("-r", "--record", help="records the video",
                            action="store_true", dest="record", default=False)
gradcam_parser.add_argument("-a", "--agent", help="agent to use", default="all",
                            choices=["all", "lrelu", "recrat", "rat"])
gradcam_parser.add_argument("--title", help="Video title the video",
                            dest="title", default=None)
gradcam_parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                            default=0, action="store", dest="seed", type=int)
