import numpy as np
import torch

from ppo import utils
from ppo.envs import make_vec_envs

def _flatten_helper(T, N, _tensor):
    return _tensor.contiguous().view(T * N, *_tensor.size()[2:])
    
def evaluate(eval_envs, actor_critic, obs_rms, env_name, seed, num_processes, sku_mini_batch, eval_log_dir,
             device):

    eval_episode_rewards = []
    
    obs = eval_envs.reset()
    sku_num = obs.shape[1]
    
    if(sku_num%sku_mini_batch == 0):
        sku_mini_batch_num = sku_num//sku_mini_batch
        sku_perm = [i for i in range(sku_num)]
        e_sku_num = sku_num
    else:
        sku_mini_batch_num = sku_num//sku_mini_batch + 1
        e_sku_num = sku_mini_batch_num*sku_mini_batch
        sku_perm = [i for i in range(sku_num)]
        sku_perm = sku_perm + [sku_perm[-1] for i in range(e_sku_num-sku_num)]
            
    eval_recurrent_hidden_states = []
    for i in range(sku_mini_batch_num):
        eval_recurrent_hidden_states.append(torch.zeros(
            sku_mini_batch*num_processes, actor_critic.recurrent_hidden_state_size, device=device))
    eval_masks = torch.zeros(sku_mini_batch*num_processes, 1, device=device)
            
    while True:
        actions = []
        with torch.no_grad():
            for i in range(sku_mini_batch_num):
            
                stack_sku_indexes = [sku_perm[i*sku_mini_batch+j] for j in range(sku_mini_batch)]
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
            for j in range(sku_mini_batch):
                sku_index = sku_perm[i*sku_mini_batch+j]
                if action_[sku_index] is None:
                    action_[sku_index] = actions[i][j*num_processes: j*num_processes+num_processes,:]
                        
        action_ = torch.stack(action_)
        action_ = action_.transpose(0, 1)
        # Obser reward and next obs
        
        obs, _, done, infos = eval_envs.step(action_)

        eval_masks = torch.tensor(
            [[0.0] if done[0] else [1.0] for i in range(sku_mini_batch*num_processes)],
            dtype=torch.float32,
            device=device)
        
        epo_temp_reward = []
        for info in infos:
            if 'episode' in info.keys():
                epo_temp_reward.append(info['episode']['r'])

        if(len(epo_temp_reward) != 0):
            eval_episode_rewards.append(np.mean(epo_temp_reward))
            break

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    return np.mean(eval_episode_rewards)