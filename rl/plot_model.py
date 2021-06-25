from tensorflow.keras.utils import plot_model

from .games import create_game
from .utils import create_parser, parse_args


def main():
    # parse args
    parser = create_parser()
    parser.add_argument('--to_file', type=str, default='model.png')

    (args, cfg, agent_cls, model_cls) = parse_args(parser)
    cfg_game = cfg.get('game', {})
    cfg_game['display'] = False
    cfg_game['episodes'] = 0
    cfg_game['episode_start'] = 0
    cfg_game['load_weights'] = False
    game = create_game(args.game, cfg, agent_cls, model_cls)
    print(f'write model file to {args.to_file}')
    plot_model(game.agent.model, to_file=args.to_file, show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    main()
