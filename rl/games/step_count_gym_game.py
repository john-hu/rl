from abc import abstractmethod
from .gym_base import GymBaseGame


class StepCountGymGame(GymBaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.step = 0
        self.done = False

    @property
    @abstractmethod
    def game_name(self):
        pass

    def get_score(self):
        return self.step

    def on_step_result(self, state, action, reward, next_state, done):
        self.step += 1
        self.done = done

    def on_game_reset(self, episode):
        self.step = 0
        self.done = False

    def on_game_end(self, episode):
        episode_index = self.updated_cfg['game'].get('episode_start', 1) + episode
        print(f'Episode {episode_index}# Score: {self.step} ==> {self.done}')
