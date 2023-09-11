import copy
import glob
import os
import time
from collections import deque
import random
import math

import numpy as np
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo import algo, utils
from arguments.rl_train_args import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy
from ppo.storage import RolloutStorage
from evaluation import evaluate

def _flatten_helper(T, N, _tensor):
    return _tensor.contiguous().view(T * N, *_tensor.size()[2:])

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(10)
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.train_demand_data_pth, args.vlt_data_pth, False, args.num_processes,
                         args.gamma, args.log_dir, device, True)
    
    eval_envs = make_vec_envs(args.env_name, args.seed, args.test_demand_data_pth, args.vlt_data_pth, False, args.num_processes,
                              None, eval_log_dir, device, True, eval = True)
    
    obs = envs.reset()
    sku_num = obs.shape[1]

    if(args.resume == None):
        actor_critic = Policy(
            envs.observation_space.shape,        
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    else:
        actor_critic, obs_rms = torch.load(args.resume)
        
    actor_critic.to(device)

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.sku_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr[0],
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    episode_rewards = deque(maxlen=10)

    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    best_reward = float("-inf")
    no_improvement_update_num = 0
    lr_index = 0
    for update_num in range(num_updates):
        
        if(sku_num < args.sku_mini_batch):
            sku_mini_batch_num = 1
            args.sku_mini_batch = sku_num
            sku_perm = [i for i in range(sku_num)]
            random.shuffle(sku_perm)
            e_sku_num = sku_num
        if(sku_num%args.sku_mini_batch == 0):
            sku_mini_batch_num = sku_num//args.sku_mini_batch
            sku_perm = [i for i in range(sku_num)]
            random.shuffle(sku_perm)
            e_sku_num = sku_num
        else:
            sku_mini_batch_num = sku_num//args.sku_mini_batch + 1
            e_sku_num = sku_mini_batch_num*args.sku_mini_batch
            sku_perm = [i for i in range(sku_num)]
            random.shuffle(sku_perm)
            sku_perm = sku_perm + [sku_perm[-1] for i in range(e_sku_num-sku_num)]
        
        rollouts = []
        for i in range(sku_mini_batch_num):
            rollouts.append(RolloutStorage(args.num_steps, args.num_processes*args.sku_mini_batch,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size))
            stack_sku_indexes = [sku_perm[i*args.sku_mini_batch+j] for j in range(args.sku_mini_batch)]
            sku_mini_batch_obs = obs[:,stack_sku_indexes,:].permute(1, 0, 2)
            sku_mini_batch_obs = _flatten_helper(sku_mini_batch_obs.shape[0], sku_mini_batch_obs.shape[1], sku_mini_batch_obs)
            rollouts[-1].obs[0].copy_(sku_mini_batch_obs)
            rollouts[-1].to(device)
                
        for step in range(args.num_steps):
            values = [None for i in range(sku_mini_batch_num)]
            actions = [None for i in range(sku_mini_batch_num)]
            action_log_probs = [None for i in range(sku_mini_batch_num)]
            recurrent_hidden_states = [None for i in range(sku_mini_batch_num)]
            for i in range(sku_mini_batch_num):
                
                with torch.no_grad():
                    values[i], actions[i], action_log_probs[i], recurrent_hidden_states[i] = actor_critic.act(
                        rollouts[i].obs[step], rollouts[i].recurrent_hidden_states[step],
                        rollouts[i].masks[step])
            action_ = [None for i in range(sku_num)]
            
            for i in range(sku_mini_batch_num):
                for j in range(args.sku_mini_batch):
                    sku_index = sku_perm[i*args.sku_mini_batch+j]
                    if action_[sku_index] is None:
                        action_[sku_index] = actions[i][j*args.num_processes: j*args.num_processes+args.num_processes,:]
            action_ = torch.stack(action_)
            action_ = action_.transpose(0, 1)
            
            obs, reward, done, infos = envs.step(action_)
            
            rewards = []
            temp_reward = []
            for info in infos:
                rewards.append(np.array(info["rewards"]))
                if 'episode' in info.keys():
                    temp_reward.append(info['episode']['r'])
            if(len(temp_reward) != 0):
                episode_rewards.append(np.mean(temp_reward))
            rewards = np.stack(rewards)
            rewards = torch.from_numpy(rewards).unsqueeze(dim=1).float()

            masks = torch.FloatTensor(
                [[0.0] if done[0] else [1.0] for i in range(args.num_processes*args.sku_mini_batch)])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in infos[0].keys() else [1.0]
                 for i in range(args.num_processes*args.sku_mini_batch)])

            for i in range(sku_mini_batch_num):

                stack_sku_indexes = [sku_perm[i*args.sku_mini_batch+j] for j in range(args.sku_mini_batch)]
                sku_mini_batch_obs = obs[:,stack_sku_indexes,:].permute(1, 0, 2)
                sku_mini_batch_obs = _flatten_helper(sku_mini_batch_obs.shape[0], sku_mini_batch_obs.shape[1], sku_mini_batch_obs)

                sku_mini_batch_reawrds_ = []
                for index in stack_sku_indexes:
                    sku_mini_batch_reawrds_.append(rewards[:,:,index])
                sku_mini_batch_reawrds_ = torch.stack(sku_mini_batch_reawrds_, dim=0)
                sku_mini_batch_reawrds_ = _flatten_helper(sku_mini_batch_reawrds_.shape[0], sku_mini_batch_reawrds_.shape[1], sku_mini_batch_reawrds_)

                rollouts[i].insert(sku_mini_batch_obs, recurrent_hidden_states[i], actions[i],
                                action_log_probs[i], values[i], sku_mini_batch_reawrds_, masks, bad_masks)
        
        value_losses = [None for i in range(sku_mini_batch_num)]
        action_losses = [None for i in range(sku_mini_batch_num)]
        dist_entropys = [None for i in range(sku_mini_batch_num)]
        
        if (args.eval_interval is not None and update_num % args.eval_interval == 0):
            obs_rms = []
            e_r = evaluate(eval_envs, actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, args.sku_mini_batch, eval_log_dir, device)
            if e_r > best_reward and args.save_dir != "":
                save_path = args.save_dir
                torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)], save_path)
                best_reward = e_r
                no_improvement_update_num = 0
            else:
                no_improvement_update_num += 1
                if(no_improvement_update_num == args.lr_decay_interval):
                    lr_index += 1
                    if(lr_index == len(args.lr)):
                        print("Training fininshes becasue of no improvement")
                        with open(args.eva_log_pth, 'a') as f:
                            f.write(args.save_dir + ' ' + str(args.seed) + ' ' + str(best_reward))
                            f.write('\n')
                        return
                    else:
                        utils.update_schedule(agent.optimizer, args.lr[lr_index])
                    no_improvement_update_num = 0


        for i in range(sku_mini_batch_num):
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts[i].obs[-1], rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1]).detach()

            rollouts[i].compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            value_losses[i], action_losses[i], dist_entropys[i] = agent.update(rollouts[i])

            rollouts[i].after_update()
        
        for rollout in rollouts:
            del rollout
        
        if update_num % args.log_interval == 0 and len(episode_rewards) > 0:
            total_num_steps = (update_num + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, lr {}\n"
                .format(update_num, total_num_steps,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), str(args.lr[lr_index])))

if __name__ == "__main__":
    main()
