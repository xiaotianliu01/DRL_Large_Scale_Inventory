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
from arguments.dl_train_args import get_args
from ppo.envs import make_vec_envs
from ppo.model import Policy
from evaluation import evaluate
from myenv.single_echelon import MyEnv

def _flatten_helper(T, N, _tensor):
    return _tensor.contiguous().view(T * N, *_tensor.size()[2:])

def collect_data(actor_critic, envs, device, args):

    obs = envs.reset()
    sku_num = obs.shape[1]

    print(sku_num)
    
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
    
    obs_set = [[] for i in range(sku_mini_batch_num)]
    optimal_actions_set = [[] for i in range(sku_mini_batch_num)]
    rewards_set = [[] for i in range(sku_mini_batch_num)]

    for i in range(sku_mini_batch_num):
        stack_sku_indexes = [sku_perm[i*args.sku_mini_batch+j] for j in range(args.sku_mini_batch)]
        sku_mini_batch_obs = obs[:,stack_sku_indexes,:].permute(1, 0, 2)
        sku_mini_batch_obs = _flatten_helper(sku_mini_batch_obs.shape[0], sku_mini_batch_obs.shape[1], sku_mini_batch_obs)
        obs_set[i].append(sku_mini_batch_obs)
    
    recurrent_hidden_states = [torch.zeros(args.sku_mini_batch*args.num_processes, actor_critic.recurrent_hidden_state_size, device=device) for i in range(sku_mini_batch_num)]
    masks = [torch.ones(args.sku_mini_batch*args.num_processes, 1, device=device) for i in range(sku_mini_batch_num)]

    for step in range(args.num_steps):
        
        actions = [None for i in range(sku_mini_batch_num)]
        values = [None for i in range(sku_mini_batch_num)]
        for i in range(sku_mini_batch_num):
            with torch.no_grad():
                values[i], actions[i], _, recurrent_hidden_states[i] = actor_critic.act(obs_set[i][-1], recurrent_hidden_states[i], masks[i], deterministic=True)
        
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
        optimal_actions = []
        for info in infos:
            rewards.append(np.array(info["rewards"]))
            optimal_actions.append(np.array(info["optimal_actions"]))
        rewards = torch.tensor(np.stack(rewards), device=device).float()
        rewards = torch.reshape(rewards, [rewards.shape[0], 1, rewards.shape[1]])
        
        optimal_actions = torch.tensor(np.stack(optimal_actions), device=device).float()
        optimal_actions = torch.reshape(optimal_actions, [optimal_actions.shape[0], 1, optimal_actions.shape[1]])

        for i in range(sku_mini_batch_num):

            stack_sku_indexes = [sku_perm[i*args.sku_mini_batch+j] for j in range(args.sku_mini_batch)]
            sku_mini_batch_obs = obs[:,stack_sku_indexes,:].permute(1, 0, 2)
            sku_mini_batch_obs = _flatten_helper(sku_mini_batch_obs.shape[0], sku_mini_batch_obs.shape[1], sku_mini_batch_obs)
            sku_mini_batch_reawrds_ = []
            sku_mini_batch_optimal_actions_ = []
            for index in stack_sku_indexes:
                sku_mini_batch_reawrds_.append(rewards[:,:,index])
                sku_mini_batch_optimal_actions_.append(optimal_actions[:,:,index])
            sku_mini_batch_reawrds_ = torch.stack(sku_mini_batch_reawrds_, dim=0)
            sku_mini_batch_reawrds_ = _flatten_helper(sku_mini_batch_reawrds_.shape[0], sku_mini_batch_reawrds_.shape[1], sku_mini_batch_reawrds_)
            sku_mini_batch_optimal_actions_ = torch.stack(sku_mini_batch_optimal_actions_, dim=0)
            sku_mini_batch_optimal_actions_ = _flatten_helper(sku_mini_batch_optimal_actions_.shape[0], sku_mini_batch_optimal_actions_.shape[1], sku_mini_batch_optimal_actions_)

            obs_set[i].append(sku_mini_batch_obs)
            optimal_actions_set[i].append(sku_mini_batch_optimal_actions_)
            rewards_set[i].append(sku_mini_batch_reawrds_)

        if(done[0]):
            break
    def stack_things(li):
        length = len(li[0])
        res = []
        for i in range(length):
            temp = []
            for j in range(len(li)):
                temp.append(li[j][i])
            temp = torch.cat(temp, dim=0) 
            res.append(temp)
        return res
    stacked_obs = stack_things(obs_set)
    stacked_rewards = stack_things(rewards_set)
    stacked_optimal_actions = stack_things(optimal_actions_set)
    return stacked_obs, stacked_rewards, stacked_optimal_actions

def dl_train_actor():
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

    envs = make_vec_envs(args.env_name, args.seed, args.train_demand_data_pth, args.vlt_data_pth, True, args.num_processes,
                         args.gamma, args.log_dir, device, True)
    eval_envs = make_vec_envs(args.env_name, args.seed, args.test_demand_data_pth, args.vlt_data_pth, False, args.num_processes,
                              None, eval_log_dir, device, True, eval = True)

    if(args.resume == None):
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    else:
        actor_critic, obs_rms = torch.load(args.resume)
        
    actor_critic.to(device)
    
    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr[0], eps=args.eps)
    best_reward = -float('inf')

    lr_index = 0
    no_improvement_update_num = 0

    print("Start training actor ...")
    for update_num in range(args.actor_epochs):
        
        obs_set, rewards_set, optimal_actions_set = collect_data(actor_critic, envs, device, args)
        sku_num = obs_set[0].shape[0]

        loss = []

        for mini_update_num in range(args.actor_mini_epochs):
            
            actions_p = []
            recurrent_hidden_states = torch.zeros(sku_num, actor_critic.recurrent_hidden_state_size, device=device)
            masks = torch.ones(sku_num, 1, device=device)
            for i in range(len(obs_set)):
                value, action, _, recurrent_hidden_states = actor_critic.act(obs_set[i], recurrent_hidden_states, masks, deterministic=True)
                actions_p.append(action)
            
            actions_p = torch.cat(actions_p[:-1], dim = 0)
            target_act = torch.cat(optimal_actions_set, dim = 0)

            updata_batch_size = int(actions_p.shape[0]/args.update_batch_num)
            for i in range(args.update_batch_num):
                action_losses = (actions_p[i*updata_batch_size:(i+1)*updata_batch_size] - target_act[i*updata_batch_size:(i+1)*updata_batch_size]).pow(2)
                loss.append(action_losses.mean().item())
                optimizer.zero_grad()
                action_losses.mean().backward()
                nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
                optimizer.step()

        if (update_num % 1 == 0):
            print("Updates ", update_num, ", Best Performance ", best_reward, ", No Improvement Updates ", no_improvement_update_num,  ", lr ", args.lr[lr_index], ", actor loss ", np.mean(loss))
            obs_rms = []
            e_r = evaluate(eval_envs, actor_critic, obs_rms, args.env_name, args.seed,
                    args.num_processes, args.sku_mini_batch, eval_log_dir, device)
            if e_r > best_reward and args.save_dir != "":
                save_path = args.save_dir
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(eval_envs), 'obs_rms', None)
                ], save_path)
                best_reward = e_r
                no_improvement_update_num = 0
            else:
                no_improvement_update_num += 1
                if(no_improvement_update_num == args.lr_decay_interval):
                    lr_index += 1
                    if(lr_index == len(args.lr)):
                        print("Training fininshes becasue of no improvement")
                        break
                    else:
                        utils.update_schedule(optimizer, args.lr[lr_index])
                    no_improvement_update_num = 0

def dl_train_critic():

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval_dummy"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(10)
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.train_demand_data_pth, args.vlt_data_pth, True, args.num_processes,
                         args.gamma, args.log_dir, device, True)

    actor_critic, obs_rms = torch.load(args.save_dir)
        
    actor_critic.to(device)

    for k,v in actor_critic.named_parameters():
        if('critic' in k):
            continue
        v.requires_grad = False

    params = filter(lambda p: p.requires_grad, actor_critic.parameters())
    value_optimizer = optim.Adam(params, lr=args.lr[0], eps=args.eps)

    obs_set, rewards_set, optimal_actions_set = collect_data(actor_critic, envs, device, args)
    sku_num = obs_set[0].shape[0]

    print("Start training critic ...")

    for value_epoch in range(args.critic_epochs):
        values_p = []
        recurrent_hidden_states = torch.zeros(sku_num, actor_critic.recurrent_hidden_state_size, device=device)
        masks = torch.ones(sku_num, 1, device=device)
        for i in range(len(obs_set)):
            value, action, _, recurrent_hidden_states = actor_critic.act(obs_set[i], recurrent_hidden_states, masks, deterministic=True)
            values_p.append(value)
        
        returns = [0 for i in range(len(values_p)-1)]
        gae = 0
        for i in reversed(range(len(returns))):
            delta = rewards_set[i] + args.gamma * values_p[i+1] - values_p[i]
            gae = delta + args.gamma * args.gae_lambda * gae
            returns[i] = gae + values_p[i]
        
        values_p = torch.cat(values_p[:-1], dim = 0)
        returns = torch.cat(returns, dim = 0)
        loss = []
        updata_batch_size = int(values_p.shape[0]/args.update_batch_num)
        for i in range(args.update_batch_num):
            value_losses = (values_p[i*updata_batch_size:(i+1)*updata_batch_size] - returns[i*updata_batch_size:(i+1)*updata_batch_size]).pow(2)
            value_optimizer.zero_grad()
            value_losses.mean().backward()
            loss.append(value_losses.mean().item())
            nn.utils.clip_grad_norm_(actor_critic.parameters(), args.max_grad_norm)
            value_optimizer.step()
        print("Epochs ", value_epoch, " , value loss ", np.mean(loss))
    
    for k,v in actor_critic.named_parameters():
        v.requires_grad = True
    torch.save([actor_critic, getattr(utils.get_vec_normalize(envs), 'obs_rms', None)], args.save_dir)

if __name__ == "__main__":
    dl_train_actor()
    dl_train_critic()
