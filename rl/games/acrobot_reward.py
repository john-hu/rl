from .total_reward_gym_game import TotalRewardGymGame


# Input:
#     * 0: apply -1 torque on the joint of 2 pendulum links
#     * 1: apply 0 torque on the joint of 2 pendulum links
#     * 2: apply +1 torque on the joint of 2 pendulum links
# Observe:
#     * 0: 1st cos(pendulum angle) (-1 to 1)
#     * 1: 1nd sin(pendulum angle) (-1 to 1)
#     * 2: 2nd cos(pendulum angle) (-1 to 1)
#     * 3: 2nd sin(pendulum angle) (-1 to 1)
#     * 4: 1st pendulum velocity (-4 pi to 4 pi)
#     * 5: 2nd pendulum velocity (-9 pi to 9 pi)
# Rewards:
#     * 0: -cos(s[0]) - cos(s[1] + s[0]) > 1
#     * -1: otherwise
# Termination:
#     * -cos(s[0]) - cos(s[1] + s[0]) > 1.
#     * step == 500
class AcrobotReward(TotalRewardGymGame):
    @property
    def game_name(self):
        return 'Acrobot-v1'

    @property
    def name(self):
        return 'Acrobot-v1-rewarded'

    def transform_reward(self, _state, _action, reward, next_state, _done):
        self.done = reward == 0
        return 200 if reward == 0 else abs(next_state[4])
