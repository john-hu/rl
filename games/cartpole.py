from .step_count_gym_game import StepCountGymGame


# Input:
#     * 0: push left
#     * 1: push right
# Observe:
#     * 0: Cart position (-4.8 ~ 4.8)
#     * 1: Cart velocity (-Inf ~ Inf)
#     * 2: Pole angle (-0.418 radius, -24deg ~ 0.418 radius 24 deg)
#     * 3: Pole angular velocity (-Inf ~ Inf)
# Rewards:
#     * 0: terminated
#     * 1: still alive
# Termination:
#     * Pole angle > 12 deg or Pole angle < -12 deg
#     * Cart position > 2.4 or Cart position < -2.4
#     * step == 500
class CartPole(StepCountGymGame):
    @property
    def game_name(self):
        return 'CartPole-v1'

    def check_replay_training(self, skip_rule):
        return self.step > skip_rule
