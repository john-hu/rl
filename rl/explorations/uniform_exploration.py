import numpy as np


class UniformExploration:
    def __init__(self, cfg):
        self.__exploration_rate = cfg.get('exploration_rate', 1.0)
        self.__exploration_min = cfg.get('exploration_min', 0.01)
        self.__exploration_decay = cfg.get('exploration_decay', 0.995)

    def is_exploring(self):
        return np.random.rand() <= self.__exploration_rate

    def decay(self):
        if self.__exploration_rate > self.__exploration_min:
            self.__exploration_rate *= self.__exploration_decay

    def set_to_min(self):
        self.__exploration_rate = self.__exploration_min
