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
    @property
    def game_name(self):
        return 'LunarLander-v2'

    def transform_reward(self, _state, _action, reward, _next_state, _done):
        self.total_reward += reward
        self.step += 1
        return self.total_reward

    def on_step_result(self, state, action, reward, next_state, done):
        if done and reward > 50:
            self.done = True

    def check_replay_training(self, skip_rule):
        return self.done
