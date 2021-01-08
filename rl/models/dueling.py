import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam


def calc_output_value(data):
    v_value = K.expand_dims(data[:, 0], -1)
    a_value = data[:, 1:]
    max_a = K.max(data[:, 1:], keepdims=True)
    return v_value + a_value - max_a


def create_dueling_model(cfg, layer_count, dense_count):
    assert layer_count > 1, 'at least one layer'
    input_dense = Input(cfg['input'])
    current_dense = input_dense
    for i in range(layer_count):
        current_dense = Dense(dense_count[i], activation='relu')(current_dense)
    dueling_dense = Dense(cfg['output'] + 1, activation='linear')(current_dense)
    output = Lambda(calc_output_value, output_shape=(cfg['output'],))(dueling_dense)
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
