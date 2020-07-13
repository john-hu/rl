from .acrobot import Acrobot
from .cartpole import CartPole
from .mountain_car import MountainCar


def create_game(game, cfg, agent_cls, model_cls):
    if game == 'acrobot':
        return Acrobot(cfg, agent_cls, model_cls)
    if game == 'cartpole':
        return CartPole(cfg, agent_cls, model_cls)
    if game == "mountain_car":
        return MountainCar(cfg, agent_cls, model_cls)
    raise f'game {game} is not supported'
