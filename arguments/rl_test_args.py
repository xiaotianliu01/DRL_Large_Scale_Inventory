import argparse

def get_args():

    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--sku-mini-batch',
        type=int,
        default=1000,
        help='sku mini batch szie')
    parser.add_argument(
        '--env-name',
        default='single_echelon',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--demand-data-pth',
        default='./data/df_sales.csv',
        help='demand data path for test')
    parser.add_argument(
        '--vlt-data-pth',
        default='./data/df_vlt.csv',
        help='vlt data path for test')
    parser.add_argument(
        '--load-dir',
        default='./models/rl_single_echelon.pt',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--test_pics_save_dir',
        default=None,
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()
    args.det = not args.non_det

    return args
