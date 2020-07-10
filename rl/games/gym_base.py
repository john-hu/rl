from abc import abstractmethod
import gym

from .base import BaseGame


class GymBaseGame(BaseGame):
    @property
    @abstractmethod
    def game_name(self):
        pass

    def create_env(self, cfg):
        self.env = gym.make(self.game_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        cfg['state_size'] = self.state_size
        cfg['action_size'] = self.action_size
        return cfg
