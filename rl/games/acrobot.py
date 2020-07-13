from .total_reward_gym_game import TotalRewardGymGame


class Acrobot(TotalRewardGymGame):
    @property
    def game_name(self):
        return 'Acrobot-v1'
