import argparse
import json
import os

from .agents import get_agent_cls
from .models import get_model_cls


def str2bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--agent', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    return parser


def parse_args(parser):
    args = parser.parse_args()
    assert os.path.exists(args.config), f'{args.config} is not existing'
    assert os.path.isfile(args.config), f'{args.config} is not a file'
    with open(args.config) as file_handle:
        cfg = json.load(file_handle)
    agent_cls = get_agent_cls(args.agent)
    model_cls = get_model_cls(args.model)
    return (args, cfg, agent_cls, model_cls)
