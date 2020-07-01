import argparse
import os
import gym

from agent import Agent

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


class RL:
  def __init__(self):
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', type=str2bool, default=False)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--training-batch-size', type=int, default=32)
    args = parser.parse_args()
    self.display = args.display
    self.training_batch_size = args.training_batch_size
    self.episodes = args.episodes
    # create env
    self.env = gym.make('CartPole-v1')
    self.state_size = self.env.observation_space.shape[0]
    self.action_size = self.env.action_space.n
    # create agent
    self.agent = Agent({
      'state_size': self.state_size,
      'action_size': self.action_size
    })
    # load weights if available
    self.weights_path = os.path.join(os.path.dirname(__file__), 'weights.h5')
    self.agent.load_weights(self.weights_path)

  def run(self):
    print(f'Run CartPole with display: {self.display} and episodes: {self.episodes}')
    try:
      for episode in range(self.episodes):
        state = self.env.reset()

        done = False
        step_count = 0
        while not done:
          if self.display:
            self.env.render()
          # choose action with the model
          action = self.agent.choose_action(state)
          (next_state, reward, done, _) = self.env.step(action)
          self.agent.remember(state, action, reward, next_state, done)
          state = next_state
          step_count += 1
        print('Episode {}# Score: {}'.format(episode + 1, step_count))
        # train the model with the history
        self.agent.replay(self.training_batch_size)
    finally:
      # always save the model even if we press ctrl+c
      self.agent.save_weights(self.weights_path)


if __name__ == '__main__':
    rl = RL()
    rl.run()
