from .gym_base import GymBaseGame


"""
Input:
    * 0: accelerate left
    * 1: no accelerate
    * 2: accelerate right
Observe:
    * 0: position (-1.2 ~ 0.6)
    * 1: velocity (-0.07 ~ 0.07)
Rewards:
    * 0: get to the goal
    * -1: still at valley
Termination:
    * step == 200
    * position > 0.5
"""
class MountainCar(GymBaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.total_reward = 0

    @property
    def game_name(self):
        return 'MountainCar-v0'

    def on_step_result(self, state, action, reward, next_state, done):
        self.total_reward += reward

    def on_game_reset(self, episode):
        self.total_reward = 0

    def on_game_end(self, episode):
        print(f'Episode {episode + 1}# Total Score: {self.total_reward}')
