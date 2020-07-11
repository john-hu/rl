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
