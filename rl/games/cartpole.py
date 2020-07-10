from .gym_base import GymBaseGame


class CartPole(GymBaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.step_count = 0

    @property
    def game_name(self):
        return 'CartPole-v1'

    def on_step_result(self, state, action, reward, next_state, done):
        self.step_count += 1

    def on_game_reset(self, episode):
        self.step_count = 0

    def on_game_end(self, episode):
        print(f'Episode {episode + 1}# Score: {self.step_count}')
