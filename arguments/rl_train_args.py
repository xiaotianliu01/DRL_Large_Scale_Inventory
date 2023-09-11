import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='ppo')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=[5e-4, 2e-4, 1e-4, 5e-5], help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=1,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=True,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--train-demand-data-pth',
        default='./data/df_sales_train.csv',
        help='demand data path for train')
    parser.add_argument(
        '--test-demand-data-pth',
        default='./data/df_sales.csv',
        help='demand data path for test')
    parser.add_argument(
        '--vlt-data-pth',
        default='./data/df_vlt.csv',
        help='vlt data path for test')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=75,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=1,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--sku-mini-batch',
        type=int,
        default=256,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=30e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='single_echelon',
        help='environment to train on')
    parser.add_argument(
        '--log-dir',
        default='./tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./models/rl_single_echelon.pt',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--resume',
        default=None,
        help='resume model path')
    parser.add_argument(
        '--eva-log-pth',
        default=None,
        help='evaluate log path')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=True,
        help='use a recurrent policy')
    parser.add_argument(
        '--lr-decay-interval',
        type=int,
        default=80,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()


    return args
