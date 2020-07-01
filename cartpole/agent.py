import os
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Agent():
  def __init__(self, cfg):
    self.state_size = cfg['state_size']
    self.action_size = cfg['action_size']
    self.memory = deque(maxlen=cfg.get('memory_size', 2000))
    self.learning_rate = cfg.get('learning_rate', 0.001)
    self.gamma = cfg.get('gamma', 0.95)
    self.exploration_rate = cfg.get('exploration_rate', 1.0)
    self.exploration_min = cfg.get('exploration_min', 0.01)
    self.exploration_decay = cfg.get('exploration_decay', 0.995)
    self.model = None
    self.__build_model()

  def __build_model(self):
    self.model = Sequential()
    self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    self.model.add(Dense(24, activation='relu'))
    self.model.add(Dense(self.action_size, activation='linear'))
    self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
    return self.model

  def load_weights(self, weights_path):
    if not os.path.isfile(weights_path):
      return
    self.model.load_weights(weights_path)
    self.exploration_rate = self.exploration_min

  def save_weights(self, weights_path):
    if not self.model:
      return
    self.model.save(weights_path)

  def choose_action(self, state):
    if np.random.rand() <= self.exploration_rate:
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
    if self.exploration_rate > self.exploration_min:
      self.exploration_rate *= self.exploration_decay
