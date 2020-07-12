import os
from abc import ABC, abstractmethod


class BaseGame(ABC):
    def __init__(self, cfg, agent_cls, model_cls):
        print('init game')
        cfg_game = cfg.get('game', {})
        self.display = cfg_game.get('display', False)
        self.training_batch_size = cfg_game.get('training_batch_size', 32)
        self.episodes = cfg_game.get('episodes', 10000)
        self.state_size = 0
        self.action_size = 0
        self.env = None
        self.agent = None
        self.train_on_step = cfg_game.get('train_on_step', False)
        self.train_on_replay = cfg_game.get('train_on_replay', True)
        # create env
        updated_cfg = self.create_env(cfg)
        print('create env done')
        self.create_agent(updated_cfg, agent_cls, model_cls)

    @abstractmethod
    def create_env(self, cfg):
        pass

    @abstractmethod
    def on_step_result(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def on_game_reset(self, episode):
        pass

    @abstractmethod
    def on_game_end(self, episode):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    def create_agent(self, cfg, agent_cls, model_cls):
        assert self.state_size > 0, 'state, env is not initialized correctly'
        assert self.action_size > 0, 'action, env is not initialized correctly'
        cfg['state_size'] = self.state_size
        cfg['action_size'] = self.action_size
        # create agent
        self.agent = agent_cls(cfg, model_cls)
        print('create agent done')
        # load weights if available
        self.weights_path = os.path.join(cfg.get('weight_path', './weights'), f'{self.name}.h5')
        self.agent.load_weights(self.weights_path)

    @abstractmethod
    def run(self):
        pass
