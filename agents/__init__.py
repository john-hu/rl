from .simple import SimpleAgent
from .simple_double_dqn import SimpleDoubleDQNAgent


def get_agent_cls(agent):
    if agent == 'simple':
        return SimpleAgent
    if agent == 'simple_double_dqn':
        return SimpleDoubleDQNAgent
    raise f'agent {agent} is not supported'
