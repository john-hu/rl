import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam


def create_dueling_model(cfg, layer_count, dense_count):
    assert layer_count > 1, 'at least one layer'
    input_dense = Input(cfg['input'])
    current_dense = input_dense
    for i in range(layer_count):
        current_dense = Dense(dense_count[i], activation='relu')(current_dense)
    dueling_dense = Dense(cfg['output'] + 1, activation='linear')(current_dense)
    output = Lambda(lambda x: K.expand_dims(x[:, 0], -1)
                    + x[:, 1:] - K.mean(x[:, 1:], keepdims=True),
                    output_shape=(cfg['output'],))(dueling_dense)
    model = Model(input_dense, output)
    model.compile(loss='mse', optimizer=Adam(lr=cfg['learning_rate']))
    return model


def create_dueling_model_factory(layer_count, dense_count):
    assert layer_count > 1, 'at least one layer'

    class DuelingDQNModel:
        @staticmethod
        def create(cfg):
            return create_dueling_model(cfg, layer_count, dense_count)
    return DuelingDQNModel
