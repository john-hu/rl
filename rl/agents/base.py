from abc import ABC, abstractmethod
from collections import deque


class BaseAgent(ABC):
    def __init__(self, cfg):
        self.state_size = cfg['state_size']
        self.action_size = cfg['action_size']
        cfg_agent = cfg.get('agent', {})
        self.memory = deque(maxlen=cfg_agent.get('memory_size', 2000))
        self.learning_rate = cfg_agent.get('learning_rate', 0.001)
        self.gamma = cfg_agent.get('gamma', 0.95)

    @abstractmethod
    def load_weights(self, weights_path):
        pass

    @abstractmethod
    def save_weights(self, weights_path):
        pass

    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def train_on_step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def replay(self, batch_size):
        pass
