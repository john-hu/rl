from .games import create_game
from .utils import create_parser, parse_args, str2bool


def main():
    # parse args
    parser = create_parser()
    parser.add_argument('--display', type=str2bool, default=False)
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--episode-start', type=int, default=1)

    (args, cfg, agent_cls, model_cls) = parse_args(parser)
    cfg_game = cfg.get('game', {})
    cfg_game['display'] = args.display
    cfg_game['episodes'] = args.episodes
    cfg_game['episode_start'] = args.episode_start
    game = create_game(args.game, cfg, agent_cls, model_cls)
    game.run()


if __name__ == '__main__':
    main()
