from .gym_base import GymBaseGame


class Acrobot(GymBaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.total_reward = 0

    @property
    def game_name(self):
        return 'Acrobot-v1'

    def on_step_result(self, state, action, reward, next_state, done):
        self.total_reward += reward

    def on_game_reset(self, episode):
        self.total_reward = 0

    def on_game_end(self, episode):
        print(f'Episode {episode + 1}# Total Score: {self.total_reward}')
