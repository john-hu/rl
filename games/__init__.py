from .acrobot import Acrobot
from .acrobot_reward import AcrobotReward
from .cartpole import CartPole
from .cartpole_reward import CartPoleReward
from .lunar_lander import LunarLander
from .lunar_lander_reward import LunarLanderReward
from .mountain_car import MountainCar
from .mountain_car_reward import MountainCarReward


GAME_MAP = {
    'acrobot': Acrobot,
    'acrobot_reward': AcrobotReward,
    'cartpole': CartPole,
    'cartpole_reward': CartPoleReward,
    'lunar_lander': LunarLander,
    'lunar_lander_reward': LunarLanderReward,
    'mountain_car': MountainCar,
    'mountain_car_reward': MountainCarReward
}


def create_game(game, cfg, agent_cls, model_cls):
    assert game in GAME_MAP, f'game {game} is not supported'
    return GAME_MAP[game](cfg, agent_cls, model_cls)
