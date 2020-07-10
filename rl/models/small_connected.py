from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class SmallConnectedModel:
    @staticmethod
    def create(cfg):
        model = Sequential()
        model.add(Dense(24, input_dim=cfg['input'], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(cfg['output'], activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=cfg['learning_rate']))
        return model
