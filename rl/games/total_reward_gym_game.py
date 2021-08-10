from abc import abstractmethod
from .gym_base import GymBaseGame


class TotalRewardGymGame(GymBaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.total_reward = 0
        self.rewards = []
        self.done = False
        self.step = 0

    @property
    @abstractmethod
    def game_name(self):
        pass

    def get_score(self):
        return self.total_reward

    def on_step_result(self, state, action, reward, next_state, done):
        self.total_reward += reward
        self.step += 1

    def on_game_reset(self, episode):
        self.total_reward = 0
        self.step = 0
        self.done = False

    def on_game_end(self, episode):
        episode_index = self.updated_cfg['game'].get('episode_start', 1) + episode
        self.rewards.append(self.total_reward)
        self.rewards = self.rewards[-100:]
        avg_rewards = sum(self.rewards) / len(self.rewards)
        print(f'Episode {episode_index}# Total Score: {self.total_reward}, avg: {avg_rewards}' +\
              f', step: {self.step}, done: {self.done}')
        if self.done:
            print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
