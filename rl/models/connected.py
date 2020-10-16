from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def create_model(cfg, layer_count, dense_count, activation='relu'):
    assert layer_count > 1, 'at least one layer'
    model = Sequential()
    for i in range(layer_count):
        if i == 0:
            model.add(Dense(dense_count[i], input_dim=cfg['input'], activation=activation))
        else:
            model.add(Dense(dense_count[i], activation=activation))
    model.add(Dense(cfg['output'], activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=cfg['learning_rate']))
    return model


def create_model_factory(layer_count, dense_count, activation='relu'):
    assert layer_count > 1, 'at least one layer'

    class ConnectedModel:
        @staticmethod
        def create(cfg):
            return create_model(cfg, layer_count, dense_count, activation)
    return ConnectedModel
