from .total_reward_gym_game import TotalRewardGymGame


# Input:
#     * 0: do nothing
#     * 1: fire left engine
#     * 2: fire main engine
#     * 3: fire right engine
# Observe:
#     * s[0]: is the horizontal coordinate
#     * s[1]: is the vertical coordinate
#     * s[2]: is the horizontal speed
#     * s[3]: is the vertical speed
#     * s[4]: is the angle
#     * s[5]: is the angular speed
#     * s[6]: 1 if first leg has contact, else 0
#     * s[7]: 1 if second leg has contact, else 0
# Rewards:
#     * a weighted score, the higher the better.
# Termination:
#     * landed: zero speed.
#     * landed: at other places or touched with speed.
#     * lander is out side of screen
class LunarLander(TotalRewardGymGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.success_count = 0

    @property
    def game_name(self):
        return 'LunarLander-v2'

    def transform_reward(self, _state, _action, reward, _next_state, _done):
        # -100 for crash is too high, change it to -3.
        return reward if reward > -100 else -3

    def on_step_result(self, state, action, reward, next_state, done):
        super().on_step_result(state, action, reward, next_state, done)
        if done:
            self.final_reward = reward
        if done and reward > 99:
            # collect continuous success for stop training
            self.success_count += 1
            self.done = True
        elif done:
            self.success_count = 0

    def skip_step_training(self, _state, _action, _reward, _next_state, _done):
        # skip training if continuous 5 success.
        return self.success_count > 0

    def check_replay_training(self, skip_rule):
        # skip training if continuous 5 success.
        return self.success_count > 5
