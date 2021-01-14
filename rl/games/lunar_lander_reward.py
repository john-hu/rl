from .lunar_lander import LunarLander


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
class LunarLanderReward(LunarLander):
    @property
    def name(self):
        return 'LunarLander-reward'

    def transform_reward(self, _state, _action, reward, next_state, done):
        new_reward = -abs(next_state[0])\
            - abs(next_state[1])\
            - abs(next_state[2])\
            - abs(next_state[3])\
            - abs(next_state[4])\
            - abs(next_state[5])\
            + ((200 - self.step) / 500)
        if done and reward > 50:
            self.done = True
            new_reward += 2000
        self.total_reward += new_reward
        self.step += 1
        return new_reward

    def check_replay_training(self, skip_rule):
        return self.done
