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
        self.memory.append((
            np.reshape(state, [1, self.state_size]),
            action,
            reward,
            np.reshape(next_state, [1, self.state_size]),
            done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            training_batch = self.memory
        else:
            training_batch = random.sample(self.memory, batch_size)

        for (state, action, reward, next_state, done) in training_batch:
            target = reward
            if not done:
                # calcuare the next action rewards
                next_reward = np.amax(self.model.predict(next_state)[0])
                # adds future reward expectation
                target = reward + self.gamma * next_reward
            # predict and train
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # decay the exploration rate
        self.exploration.decay()

    def train_on_step(self, state, action, reward, next_state, done):
        pass
