import argparse
import yaml


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        default='configs/default.yaml', nargs='?')

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_epoch', type=int)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--focal_loss', type=bool)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--fusion', action='store_true')

    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb', choices=["rgb", "flow"])
    parser.add_argument('--new_length', default=64, type=int, metavar='N', help='length of sampled video frames (default: 1)')
    parser.add_argument('--iter_size', type=int, default=1)

    args = parser.parse_args()

    with open(args.config_file, 'r', encoding='utf-8') as f:
        tmp = f.read()
        data = yaml.load(tmp, Loader=yaml.FullLoader)

    data['training']['learning_rate'] = float(data['training']['learning_rate'])
    data['training']['weight_decay'] = float(data['training']['weight_decay'])

    if args.batch_size is not None:
        data['training']['batch_size'] = int(args.batch_size)
    if args.learning_rate is not None:
        data['training']['learning_rate'] = float(args.learning_rate)
    if args.weight_decay is not None:
        data['training']['weight_decay'] = float(args.weight_decay)
    if args.max_epoch is not None:
        data['training']['max_epoch'] = int(args.max_epoch)
    if args.checkpoint_path is not None:
        data['training']['checkpoint_path'] = args.checkpoint_path
        data['testing']['checkpoint_path'] = args.checkpoint_path
    if args.seed is not None:
        data['training']['random_seed'] = args.seed
    if args.focal_loss is not None:
        data['training']['focal_loss'] = args.focal_loss
    data['training']['resume'] = args.resume
    data['testing']['fusion'] = args.fusion
    if args.beta is not None:
        data['training']['beta'] = args.beta
    if args.modality is not None:
        data['training']['modality'] = args.modality
    if args.new_length is not None:
        data['training']['new_length'] = args.new_length
    if args.iter_size is not None:
        data['training']['iter_size'] = args.iter_size

    return data


config = get_config()
