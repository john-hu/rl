import numpy as np

from .simple import SimpleAgent


class SimpleDoubleDQNAgent(SimpleAgent):
    def __init__(self, cfg, model_cls):
        super().__init__(cfg, model_cls)
        cfg_agent = cfg.get('agent', {})
        self.target_model_update_interval = cfg_agent.get('target_update_interval', 100)
        self.target_model = model_cls.create({
            'input': self.state_size,
            'output': self.action_size,
            'learning_rate': self.learning_rate
        })
        print('create target model done')

    def load_weights(self, weights_path):
        super().load_weights(weights_path)
        print('init target model with predict model')
        self.__sync_predict_target_weights()

    def __calc_training_target(self, state, action, reward, next_state, done):
        target_f = super().__calc_training_target(state, action, reward, next_state, done)
        if done:
            return target_f
        # reshape the state to [1, self.state_size]
        np_next_state = np.reshape(next_state, [1, self.state_size])
        # calcuare the next action rewards with the target model
        next_reward = np.amax(self.target_model.predict(np_next_state)[0])
        # change the target_f with the next_reward
        target_f[0][action] = reward + self.gamma * next_reward
        return target_f

    def __sync_predict_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def game_end(self, episode):
        executed_count = episode + 1
        if executed_count % self.target_model_update_interval == 0 and episode > 0:
            print(f'update weights at {executed_count}')
            self.__sync_predict_target_weights()
