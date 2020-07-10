from .simple import SimpleAgent


def get_agent_cls(agent):
    if agent == 'simple':
        return SimpleAgent
    raise f'agent {agent} is not supported'
