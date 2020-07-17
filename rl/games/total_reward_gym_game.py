from abc import abstractmethod
from .gym_base import GymBaseGame


class TotalRewardGymGame(GymBaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.total_reward = 0

    @property
    @abstractmethod
    def game_name(self):
        pass

    def on_step_result(self, state, action, reward, next_state, done):
        self.total_reward += reward

    def on_game_reset(self, episode):
        self.total_reward = 0

    def on_game_end(self, episode):
        episode_index = self.updated_cfg['game'].get('episode_start', 1) + episode
        print(f'Episode {episode_index}# Total Score: {self.total_reward}')
