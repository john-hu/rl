import os
import random
import numpy as np

from ..explorations.uniform_exploration import UniformExploration
from .base import BaseAgent


class SimpleAgent(BaseAgent):
    def __init__(self, cfg, model_cls):
        super().__init__(cfg)
        self.exploration = UniformExploration(cfg)
        print('create exploration done')
        self.model = model_cls.create({
            'input': self.state_size,
            'output': self.action_size,
            'learning_rate': self.learning_rate
        })
        print('create model done')

    def load_weights(self, weights_path):
        if not os.path.isfile(weights_path):
            return
        print(f'model loaded from {weights_path}')
        self.model.load_weights(weights_path)
        self.exploration.set_to_min()

    def save_weights(self, weights_path):
        if not self.model:
            return
        self.model.save(weights_path)

    def choose_action(self, state):
        if self.exploration.is_exploring():
            return np.random.choice(self.action_size)
        np_state = np.reshape(state, [1, self.state_size])
        actions_rewards = self.model.predict(np_state)
        return np.argmax(actions_rewards[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            training_batch = self.memory
        else:
            training_batch = random.sample(self.memory, batch_size)

        for (state, action, reward, next_state, done) in training_batch:
            self.train_on_step(state, action, reward, next_state, done)
        # decay the exploration rate
        self.exploration.decay()

    def replay_batch(self, batch_size, epochs=1):
        if len(self.memory) < batch_size:
            training_batch = self.memory
        else:
            training_batch = random.sample(self.memory, batch_size)

        avg_loss = self.replay_batch_records(training_batch, epochs)
        # decay the exploration rate
        self.exploration.decay()
        if self.verbose_mode:
            print(f'avg_loss = {avg_loss}, exploration_rate = {self.exploration.exploration_rate}')
        return avg_loss

    def replay_batch_records(self, training_batch, epochs=1):
        new_state = []
        new_target_f = []

        for (state, action, reward, next_state, done) in training_batch:
            target_f = self.calc_training_target(state, action, reward, next_state, done)
            new_state.append(state)
            new_target_f.append(target_f[0])

        # train in batch
        fit = self.model.fit(np.array(new_state), np.array(
            new_target_f), epochs=epochs, verbose=0)
        avg_loss = np.average(np.array(fit.history['loss']))
        return avg_loss

    def calc_training_target(self, state, action, reward, next_state, done):
        # reshape the state to [1, self.state_size]
        np_state = np.reshape(state, [1, self.state_size])
        np_next_state = np.reshape(next_state, [1, self.state_size])
        target = reward
        if not done:
            # calcuare the next action rewards
            next_reward = np.amax(self.model.predict(np_next_state)[0])
            # adds future reward expectation
            target = reward + self.gamma * next_reward
        # predict and train
        target_f = self.model.predict(np_state)
        target_f[0][action] = target
        return target_f

    def train_on_step(self, state, action, reward, next_state, done):
        target_f = self.calc_training_target(state, action, reward, next_state, done)
        np_state = np.reshape(state, [1, self.state_size])
        self.model.fit(np_state, target_f, epochs=1, verbose=0)
