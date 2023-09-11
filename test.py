import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch

from ppo.envs import VecPyTorch, make_vec_envs
from ppo.utils import get_render_func, get_vec_normalize
from arguments.rl_test_args import get_args

def _flatten_helper(T, N, _tensor):
    return _tensor.contiguous().view(T * N, *_tensor.size()[2:])

def rl_test(args):

    torch.set_num_threads(1)
    location = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(location)

    env = make_vec_envs(
        args.env_name,
        args.seed,
        args.demand_data_pth,
        args.vlt_data_pth,
        False,
        1,
        None,
        None,
        device=device,
        allow_early_resets=True,
        eval = True,
        draw = args.test_pics_save_dir)

    # We need to use the same statistics for normalization as used in training
    actor_critic, obs_rms = \
                torch.load(args.load_dir, map_location=location)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    test_rewards = []
    obs = env.reset()
    sku_num = obs.shape[1]
    
    if(sku_num%args.sku_mini_batch == 0):
        sku_mini_batch_num = sku_num//args.sku_mini_batch
        sku_perm = [i for i in range(sku_num)]
        e_sku_num = sku_num
    else:
        sku_mini_batch_num = sku_num//args.sku_mini_batch + 1
        e_sku_num = sku_mini_batch_num*args.sku_mini_batch
        sku_perm = [i for i in range(sku_num)]
        sku_perm = sku_perm + [sku_perm[-1] for i in range(e_sku_num-sku_num)]
            
    eval_recurrent_hidden_states = []
    for i in range(sku_mini_batch_num):
        eval_recurrent_hidden_states.append(torch.zeros(
            args.sku_mini_batch, actor_critic.recurrent_hidden_state_size, device=device))
    eval_masks = torch.zeros(args.sku_mini_batch, 1, device=device)

    episode_rewards = []
    
    while True:
        actions = []
        with torch.no_grad():
            for i in range(sku_mini_batch_num):
            
                stack_sku_indexes = [sku_perm[i*args.sku_mini_batch+j] for j in range(args.sku_mini_batch)]
                sku_mini_batch_obs = obs[:,stack_sku_indexes,:].permute(1, 0, 2)
                sku_mini_batch_obs = _flatten_helper(sku_mini_batch_obs.shape[0], sku_mini_batch_obs.shape[1], sku_mini_batch_obs)
                _, action, _, eval_recurrent_hidden_states[i] = actor_critic.act(
                    sku_mini_batch_obs,
                    eval_recurrent_hidden_states[i],
                    eval_masks,
                    deterministic=True)
                actions.append(action)
        
        action_ = [None for i in range(sku_num)]
        for i in range(sku_mini_batch_num):
            for j in range(args.sku_mini_batch):
                sku_index = sku_perm[i*args.sku_mini_batch+j]
                if action_[sku_index] is None:
                    action_[sku_index] = actions[i][j,:]
        
        action_ = torch.stack(action_)
        action_ = action_.contiguous().view(1, action_.shape[0], action_.shape[1])
        # Obser reward and next obs
        obs, reward, done, infos = env.step(action_)
        for info in infos:
            episode_rewards.append(np.array(info["rewards"]))
        
        if done:
            break
        eval_masks = torch.tensor(
            [[0.0] if done else [1.0] for i in range(args.sku_mini_batch)],
            dtype=torch.float32,
            device=device)
    
    test_rewards = []
    for i in range(sku_num):
        temp = []
        for rewards in episode_rewards:
            temp.append(rewards[i])
        test_rewards.append(np.sum(temp))

    print(" Test using {} episodes: mean reward {:.5f}, std reward {:.5f}, max reward {:.5f}, min reward {:.5f}\n".format(
            len(test_rewards), np.mean(test_rewards), np.std(test_rewards), np.max(test_rewards), np.min(test_rewards)))
    
if __name__ == '__main__':

    sys.path.append('ppo')
    args = get_args()
    rl_test(args)
