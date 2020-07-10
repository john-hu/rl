import os
from abc import ABC, abstractmethod


class BaseGame(ABC):
    def __init__(self, cfg, agent_cls, model_cls):
        print('init game')
        self.display = cfg.get('display', False)
        self.training_batch_size = cfg.get('training_batch_size', 32)
        self.episodes = cfg.get('episodes', 10000)
        self.state_size = 0
        self.action_size = 0
        self.env = None
        self.agent = None
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

    def run(self):
        print(f'Run {self.name} with display: {self.display} and episodes: {self.episodes}')
        try:
            for episode in range(self.episodes):
                state = self.env.reset()
                self.on_game_reset(episode)

                done = False
                while not done:
                    if self.display:
                        self.env.render()
                    # choose action with the model
                    action = self.agent.choose_action(state)
                    (next_state, reward, done, _) = self.env.step(action)
                    self.agent.remember(state, action, reward, next_state, done)
                    self.on_step_result(state, action, reward, next_state, done)
                    state = next_state
                self.on_game_end(episode)
                # train the model with the history
                self.agent.replay(self.training_batch_size)
        finally:
            # always save the model even if we press ctrl+c
            self.agent.save_weights(self.weights_path)
