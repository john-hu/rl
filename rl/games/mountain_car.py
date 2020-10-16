from .total_reward_gym_game import TotalRewardGymGame


# Input:
#     * 0: accelerate left
#     * 1: no accelerate
#     * 2: accelerate right
# Observe:
#     * 0: position (-1.2 ~ 0.6)
#     * 1: velocity (-0.07 ~ 0.07)
# Rewards:
#     * 0: get to the goal
#     * -1: still at valley
# Termination:
#     * step == 200
#     * position > 0.5
class MountainCar(TotalRewardGymGame):
    @property
    def game_name(self):
        return 'MountainCar-v0'

    def on_step_result(self, state, action, reward, next_state, done):
        super().on_step_result(state, action, reward, next_state, done)
        self.done = done
