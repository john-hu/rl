from .cartpole import CartPole


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
class CartPoleReward(CartPole):
    @property
    def name(self):
        return 'CartPole-rewarded'

    def transform_reward(self, state, _action, reward, _next_state, _done):
        position_reward = self.env.x_threshold - abs(state[0])
        pole_reward = self.env.theta_threshold_radians - abs(state[2])
        return reward * (position_reward + pole_reward)
