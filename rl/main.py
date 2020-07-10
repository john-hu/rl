import argparse
import json
import os

from .agents import get_agent_cls
from .games import create_game
from .models import get_model_cls


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', type=str2bool, default=False)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    assert os.path.exists(args.config), f'{args.config} is not existing'
    assert os.path.isfile(args.config), f'{args.config} is not a file'
    with open(args.config) as file_handle:
        cfg = json.load(file_handle)
    agent_cls = get_agent_cls(args.agent)
    model_cls = get_model_cls(args.model)
    cfg['display'] = args.display
    cfg['episodes'] = args.episodes
    game = create_game(args.game, cfg, agent_cls, model_cls)
    game.run()


if __name__ == '__main__':
    main()
