from .small_connected import SmallConnectedModel


def get_model_cls(model):
    if model == 'small':
        return SmallConnectedModel
    raise f'model {model} is not supported'
