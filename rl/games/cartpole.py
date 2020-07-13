from .gym_base import GymBaseGame


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
