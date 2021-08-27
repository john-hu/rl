from abc import abstractmethod
import gym

from .base import BaseGame


class GymBaseGame(BaseGame):
    def __init__(self, cfg, agent_cls, model_cls):
        super().__init__(cfg, agent_cls, model_cls)
        self.skip_replay_count = 0
        self.best_buckets = []
        self.best_replay_count = self.train_on_replay.get('best_replay', 0)
        self.current_hitory = None

    @property
    @abstractmethod
    def game_name(self):
        pass

    @abstractmethod
    def check_replay_training(self, skip_rule):
        pass

    @abstractmethod
    def get_score(self):
        pass

    def skip_step_training(self, _state, _action, _reward, _next_state, _done):
        pass

    def skip_replay_training(self):
        game_cfg = self.updated_cfg.get('game', {})
        skip_rule = game_cfg.get('skip_replay_rule')
        if not skip_rule:
            return False
        if self.check_replay_training(skip_rule):
            self.skip_replay_count += 1
            print(f'skip replay training skipped count: {self.skip_replay_count}')
            # skip the training only if the count exceeded skip_replay_min_count continuously
            return self.skip_replay_count > game_cfg.get('skip_replay_min_count', 5)
        # reset the counter if one round failed
        if self.skip_replay_count > 0:
            print(f'reset skip replay training, original: {self.skip_replay_count}')
        self.skip_replay_count = 0
        return False

    def create_env(self, cfg):
        self.env = gym.make(self.game_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        cfg['state_size'] = self.state_size
        cfg['action_size'] = self.action_size
        return cfg

    def run_one_game(self, episode):
        state = self.env.reset()
        self.on_game_reset(episode)
        self.agent.game_start(episode)
        # current history for best history replay
        current_history = []
        done = False
        while not done:
            if self.display:
                self.env.render()
            # choose action with the model
            action = self.agent.choose_action(state)
            (next_state, reward, done, _) = self.env.step(action)
            reward = self.transform_reward(state, action, reward, next_state, done)
            if self.best_replay_count > 0:
                current_history.append((state, action, reward, next_state, done))
            self.agent.remember(state, action, reward, next_state, done)
            self.on_step_result(state, action, reward, next_state, done)
            if self.train_on_step and\
                    not self.skip_step_training(state, action, reward, next_state, done):
                if self.train_on_step == "step":
                    self.agent.train_on_step(state, action, reward, next_state, done)
                elif self.train_on_step == "replay":
                    self.agent.replay_batch(self.training_batch_size, 1)

            state = next_state
        self.on_game_end(episode)
        # update best bucket
        self.update_best_bucket(self.get_score(), current_history)
        self.replay_training()
        self.replay_best_buckets()
        self.agent.game_end(episode)

    def update_best_bucket(self, score, current_history):
        self.best_buckets.append({
            'score': score,
            'history': current_history
        })
        self.best_buckets.sort(key=lambda bucket: bucket['score'], reverse=True)
        self.best_buckets = self.best_buckets[0:self.best_replay_count]

    def replay_training(self):
        if self.train_on_replay['enabled'] and not self.skip_replay_training():
            batch_replay = self.train_on_replay.get('batch_replay', False)
            # train the model with the history
            if batch_replay:
                epochs = self.train_on_replay.get('batch_epochs', 1)
                self.agent.replay_batch(self.training_batch_size, epochs)
            else:
                self.agent.replay(self.training_batch_size)

    def replay_best_buckets(self):
        if self.best_replay_count < 1:
            return
        # replay the best game history
        best_scores = ', '.join([str(round(bucket['score'], 4)) for bucket in self.best_buckets])
        print(f'replay the best game history: {best_scores}')
        for bucket in self.best_buckets:
            self.agent.replay_batch_records(bucket['history'], 5)

    # pylint: disable=R0201
    def transform_reward(self, _state, _action, reward, _next_state, _done):
        return reward

    def on_step_result(self, state, action, reward, next_state, done):
        pass

    def on_game_reset(self, episode):
        pass

    def on_game_end(self, episode):
        pass
