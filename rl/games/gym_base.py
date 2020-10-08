from abc import abstractmethod
import gym

from .base import BaseGame


class GymBaseGame(BaseGame):
    @property
    @abstractmethod
    def game_name(self):
        pass

    def skip_step_training(self, _state, _action, _reward, _next_state, _done):
        pass

    def skip_replay_training(self):
        pass

    def create_env(self, cfg):
        self.env = gym.make(self.game_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        cfg['state_size'] = self.state_size
        cfg['action_size'] = self.action_size
        return cfg

    def run_one_game(self, episode):
        state = self.env.reset()
        self.on_game_reset(episode)

        done = False
        while not done:
            if self.display:
                self.env.render()
            # choose action with the model
            action = self.agent.choose_action(state)
            (next_state, reward, done, _) = self.env.step(action)
            reward = self.transform_reward(state, action, reward, next_state, done)
            self.agent.remember(state, action, reward, next_state, done)
            self.on_step_result(state, action, reward, next_state, done)
            if self.train_on_step and\
                    not self.skip_step_training(state, action, reward, next_state, done):
                self.agent.train_on_step(state, action, reward, next_state, done)
            state = next_state
        self.on_game_end(episode)
        if self.train_on_replay['enabled'] and not self.skip_replay_training():
            batch_replay = self.train_on_replay.get('batch_replay', False)
            # train the model with the history
            if batch_replay:
                epochs = self.train_on_replay.get('batch_epochs', 1)
                self.agent.replay_batch(self.training_batch_size, epochs)
            else:
                self.agent.replay(self.training_batch_size)

    # pylint: disable=R0201
    def transform_reward(self, _state, _action, reward, _next_state, _done):
        return reward

    def on_step_result(self, state, action, reward, next_state, done):
        pass

    def on_game_reset(self, episode):
        pass

    def on_game_end(self, episode):
        pass
