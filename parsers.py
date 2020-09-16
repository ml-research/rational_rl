import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-af", "--act", help="Activation function to use",
                    action="store", dest="act_f",
                    choices=['paus', 'rpau', 'lrelu', 'relu', 'silu',
                             'd+silu'])
parser.add_argument("-alg", "--algo", help="Activation function to use",
                    action="store", dest="algo",
                    choices=['DQN', 'DDQN'])
parser.add_argument("-g", "--game", help="Game to train on",
                    action="store", dest="game")
parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                    action="store", dest="seed", type=int)
parser.add_argument("--freeze", help="Freeze pau that is then non learnable",
                    action="store_true", dest="freeze_pau", default=False)
parser.add_argument("--recover", help="Recover from the last trained agent",
                    action="store_true", dest="recover", default=False)
parser.add_argument("--load", help="Load a pretrained AF from a folder",
                    action="store_true", dest="load", default=False)

graph_parser = argparse.ArgumentParser()
graph_parser.add_argument("-g", "--game", help="Game to train on",
                          action="store", dest="game")
graph_parser.add_argument("-s", "--store", help="Save the graph in svg file (instead of displaying)",
                          action="store_true", dest="store")

rendering_parser = argparse.ArgumentParser()
rendering_parser.add_argument("-a", "--act", help="Activation function to use",
                              action="store", dest="act_f",
                              choices=['paus', 'rpau', 'lrelu', 'relu',
                                       'freezed_pau'])
rendering_parser.add_argument("-alg", "--algo", help="Activation function to use",
                              action="store", dest="algo",
                              choices=['DQN', 'DDQN'])
rendering_parser.add_argument("-g", "--game", help="Game to train on",
                              action="store", dest="game")
rendering_parser.add_argument("-s", "--seed", help="Seed for pytorch + env",
                              action="store", dest="seed", type=int)
rendering_parser.add_argument("--freeze_pau", help="Freeze pau that is then non learnable",
                              action="store_true", dest="freeze_pau", default=False)
rendering_parser.add_argument("-r", "--record", help="records the video",
                              action="store_true", dest="record", default=False)
rendering_parser.add_argument("-nd", "--no_display", help="Avoid displaying",
                              action="store_true", dest="no_display", default=False)
rendering_parser.add_argument("--video_title", help="Video title the video",
                              dest="video_title", default=None)
rendering_parser.add_argument("-e", "--epoch", help="Epoch to use",
                              dest="epoch", default="500")
