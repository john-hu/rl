from .connected import create_model_factory
from .dueling import create_dueling_model_factory


def get_model_cls(model):
    if model == 'small':
        return create_model_factory(2, [24] * 2)
    if model == 'small2':
        return create_model_factory(3, [16] * 3)
    if model == 'wide':
        return create_model_factory(2, [1024] * 2)
    if model == 'deep':
        return create_model_factory(16, [24] * 16)
    if model == 'large':
        return create_model_factory(16, [1024] * 16)
    if model == 'dueling_small':
        return create_dueling_model_factory(2, [24] * 2)
    raise f'model {model} is not supported'
