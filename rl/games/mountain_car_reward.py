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
class MountainCarReward(TotalRewardGymGame):
    @property
    def game_name(self):
        return 'MountainCar-v0'

    @property
    def name(self):
        return 'MC-v0-rewarded'

    def transform_reward(self, state, _action, reward, _next_state, _done):
        return 200 if state[0] >= 0.5 else abs(state[1])

    def on_step_result(self, state, action, reward, next_state, done):
        super().on_step_result(state, action, reward, next_state, done)
        self.done = self.step < 200 and done

    def check_replay_training(self, skip_rule):
        return self.done
