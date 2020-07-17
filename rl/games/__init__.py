from .acrobot import Acrobot
from .cartpole import CartPole
from .cartpole_reward import CartPoleReward
from .mountain_car import MountainCar


GAME_MAP = {
    'acrobot': Acrobot,
    'cartpole': CartPole,
    'cartpole_reward': CartPoleReward,
    'mountain_car': MountainCar
}


def create_game(game, cfg, agent_cls, model_cls):
    assert game in GAME_MAP, f'game {game} is not supported'
    return GAME_MAP[game](cfg, agent_cls, model_cls)
