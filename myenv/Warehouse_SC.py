import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as dc
import time
from scipy.interpolate import interp1d

GOODS_FEATURE = 'CE' # CE or Ordinal or One Hot
STARTUP_ORDER = 0

GOODS_FEAT_DIM = 1

EPOSIDE_LEN = 75
B = 1
H = 1
STARTUP_ORDER_W = 0
STARTUP_ORDER_w1 = 0
STARTUP_ORDER_w2 = 0

W_M_L = 3
w1_M_L = 5
w1_W_L = 1
w2_W_L = 1
w2_M_L = 5

W_max = 60000
w_max = 30000

w2_H_C = 2
w1_H_C = 2
W_H_C = 0.5
w1_LOST_SALES_C = 2
w2_LOST_SALES_C = 2
W_LOST_SALES_C = 2
W_ORDER_COST = 0
w_ORDER_COST = 0

TRAIN_RATIO = 0.7

def interpolate_with_trues(arr, original_length, target_length):
    inter_seq = interp1d([i for i in range(original_length)], arr, kind = 'linear')
    indexes = list(np.linspace(0, original_length-1, target_length - original_length + 1)) + [i for i in range(original_length)]
    indexes.sort()
    inter_res = inter_seq(indexes)
    return inter_res

def extend_seq(seq, original_length, target_length):

    if(target_length < original_length):
        start_index = np.random.randint(0, seq.shape[1]-target_length)
        e_seq = seq[:,start_index:start_index+target_length]
    elif(seq.shape[1] == original_length):
        start_index = 0
        e_seq = seq[:,start_index:start_index+original_length]
    else:
        start_index = np.random.randint(0, seq.shape[1]-original_length)
        e_seq = seq[:,start_index:start_index+original_length]

    inter_res = interpolate_with_trues(e_seq, e_seq.shape[1], target_length).astype(np.int)

    return inter_res

class MyEnv(gym.Env):

    def __init__(self, seed, demand_data_pth, vlt_data_pth, sku_batch_id = 0, sku_parallel_batch_num = 1, eval = False, draw = None, pretrain = False):
        self.seed(seed + sku_batch_id)
        self.pretrain = pretrain
        self.goods_data_pth = demand_data_pth 
        self.read_goods_data()
        high_action = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        low_action = np.array([0.00, 0.00, 0.00, 0.00, 0.00])
        self.action_space = spaces.Box(low_action, high_action, dtype=np.float32)
        
        high_obs = np.array([np.finfo(np.float32).max for i in range(W_M_L+w1_M_L+w2_M_L+12)])
        
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.eval = eval
        self.draw = draw

    def read_goods_data(self):
        df = pd.read_csv(self.goods_data_pth)
        df_data = np.array(df)
        self.eval_goods_demand = []
        self.goods_demand = []
        for tra in df_data:
            train_val_split_index_demand = int(len(tra[1:])*TRAIN_RATIO)
            self.goods_demand.append(tra[1:train_val_split_index_demand+1])
            self.eval_goods_demand.append(tra[train_val_split_index_demand+1:])
            
        self.goods_demand = np.array(self.goods_demand)
        self.eval_goods_demand = np.array(self.eval_goods_demand)
        self.goods_num = int(self.goods_demand.shape[0])

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def normalize(self, normalizer, arr = [], ele = 0, batch = False):

        if(batch == False):
            if(len(arr) == 0):
                if(normalizer[1] != 0):
                    return (ele-normalizer[0])/normalizer[1]
                else:
                    return ele-normalizer[0]
            
            res = []
            for i in arr:
                if(normalizer[1] != 0):
                    res.append((i-normalizer[0])/normalizer[1])
                else:
                    res.append(i-normalizer[0])
            return res
        else:
            if(len(arr.shape) == 1):
                norm_mean = normalizer[:,0]
                norm_std = np.where(normalizer[:,1] != 0, normalizer[:,1], 1)
                return (arr-norm_mean)/norm_std
            else:
                norm_mean = np.expand_dims(normalizer[:,0], -1)
                norm_std = np.expand_dims(np.where(normalizer[:,1] != 0, normalizer[:,1], 1), -1)
                return (arr-norm_mean)/norm_std
    
    def demormalize(self, normalizer, ele):
        return ele*normalizer[1] + normalizer[0]
    
    def draw_pics(self):
        
        def get_seq(arr, sku_id):
            res = []
            for i in arr:
                res.append(i[sku_id])
            return res
                           
        for i in range(self.goods_num):
            plt.figure(figsize = (10, 10))
            times = [ii for ii in range(len(self.state_his["inventory_W"]))]
            plt.plot(times, get_seq(self.state_his["inventory_w1"], i), label = 'I w1')
            plt.plot(times, get_seq(self.state_his["inventory_w2"], i), label = 'I w2')
            plt.plot(times, get_seq(self.state_his["inventory_W"], i), label = 'I W')
            plt.plot(times, get_seq(self.state_his["demand_w1"], i), label = 'D w1')
            plt.plot(times, get_seq(self.state_his["demand_w2"], i), label = 'D w2')
            plt.plot(times, get_seq(self.state_his["order_w1_M"], i), label = 'O w1 M')
            plt.plot(times, get_seq(self.state_his["order_w1_W"], i), label = 'O w1 W')
            plt.plot(times, get_seq(self.state_his["order_w2_M"], i), label = 'O w2 M')
            plt.plot(times, get_seq(self.state_his["order_w2_W"], i), label = 'O w2 W')
            plt.plot(times, get_seq(self.state_his["backlog_w1"], i), label = 'B w1')
            plt.plot(times, get_seq(self.state_his["backlog_w2"], i), label = 'B w2')
            plt.plot(times, get_seq(self.state_his["backlog_w1_W"], i), label = 'B w1 W')
            plt.plot(times, get_seq(self.state_his["backlog_w2_W"], i), label = 'B w2 W')
            plt.legend()
            plt.savefig(self.draw + str(i) + '.png')

    def reset(self):

        self.inventory_W = np.zeros([self.goods_num])
        self.inventory_w1 = np.zeros([self.goods_num])
        self.inventory_w2 = np.zeros([self.goods_num])
        self.action = np.zeros([self.goods_num, 5])
        
        self.backlog_w1 = np.zeros([self.goods_num])
        self.backlog_w2 = np.zeros([self.goods_num])
        self.backlog_W = np.zeros([self.goods_num, 2])
        
        self.order_W = np.ones([self.goods_num, W_M_L])*STARTUP_ORDER_W
        self.order_w1 = np.ones([self.goods_num, w1_M_L])*STARTUP_ORDER_w1
        self.order_w2 = np.ones([self.goods_num, w2_M_L])*STARTUP_ORDER_w2

        if(self.eval):
            self.demands_w1 = extend_seq(self.eval_goods_demand, self.eval_goods_demand.shape[1], EPOSIDE_LEN+1)
            self.demands_w2 = extend_seq(self.eval_goods_demand, self.eval_goods_demand.shape[1], EPOSIDE_LEN+1)
        else:    
            self.demands_w1 = extend_seq(self.goods_demand, self.eval_goods_demand.shape[1], EPOSIDE_LEN+1)
            self.demands_w2 = extend_seq(self.goods_demand, self.eval_goods_demand.shape[1], EPOSIDE_LEN+1)
        self.normalizer = np.array([np.array([np.mean(d), np.std(d)]) for d in self.goods_demand])
        self.step_num = 1

        if(self.draw):
            self.state_his = {"inventory_W":[], "inventory_w1":[], "inventory_w2":[], "order_w1_M":[], "order_w1_W":[], "order_w2_M":[], "order_w2_W":[],"order_W":[],"demand_w1":[], "demand_w2":[], "backlog_w1":[], "backlog_w2":[], "backlog_w1_W":[], "backlog_w2_W":[], "lost_w1":[], "lost_w2":[], "lost_W":[], "occ_W":[], "occ_w1":[], "occ_w2":[], 'out_w1':[], 'out_w2':[], 'out_W':[]}
        
        state_temp_1 = np.stack([self.demands_w1[:,0], self.demands_w2[:,0], self.backlog_w1, self.backlog_w2, self.backlog_W[:,0], self.backlog_W[:,1], self.inventory_W, self.inventory_w1, self.inventory_w2], axis = 1)
        state_temp_2 = np.concatenate([state_temp_1, self.order_W, self.order_w1, self.order_w2], axis=1)
        state_temp_3 = np.stack([np.ones([self.goods_num])*np.sum(self.inventory_W)/W_max, np.ones([self.goods_num])*np.sum(self.inventory_w1)/w_max, np.ones([self.goods_num])*np.sum(self.inventory_w2)/w_max], axis=1)
        self.state = np.concatenate([self.normalize(self.normalizer, arr = state_temp_2, batch=True), state_temp_3], axis=1)
        
        return np.array(self.state)
    
    def step(self, actions, true_action = False):      
        # W->M w1->M w2->M w1->W w2->W
        if(true_action == False):
            self.action = actions # W->M w1->M w2->M w1->W w2->W
            de_actions = np.zeros_like(self.action)
            for i in range(self.action.shape[0]):
                for j in range(self.action.shape[1]):
                    de_actions[i][j] = int(np.max([int(self.demormalize(self.normalizer[i], self.action[i][j])), 0]))
            self.action = de_actions
        else:
            self.action = actions
        
        r = self.state_update()

        state_temp_1 = np.stack([self.demands_w1[:,self.step_num-1], self.demands_w2[:,self.step_num-1], self.backlog_w1, self.backlog_w2, self.backlog_W[:,0], self.backlog_W[:,1], self.inventory_W, self.inventory_w1, self.inventory_w2], axis = 1)
        state_temp_2 = np.concatenate([state_temp_1, self.order_W, self.order_w1, self.order_w2], axis=1)
        state_temp_3 = np.stack([np.ones([self.goods_num])*np.sum(self.inventory_W)/W_max, np.ones([self.goods_num])*np.sum(self.inventory_w1)/w_max, np.ones([self.goods_num])*np.sum(self.inventory_w2)/w_max], axis=1)
        self.state = np.concatenate([self.normalize(self.normalizer, arr = state_temp_2, batch=True), state_temp_3], axis=1)
        
        if(self.step_num == EPOSIDE_LEN):
            done = True
        else:
            done = False
        
        if(self.draw):
            self.state_his["inventory_W"].append(dc(self.inventory_W))
            self.state_his["inventory_w1"].append(dc(self.inventory_w1))
            self.state_his["inventory_w2"].append(dc(self.inventory_w2))
            self.state_his["order_w1_W"].append(dc(self.action[:,3]))
            self.state_his["order_w1_M"].append(dc(self.action[:,1]))
            self.state_his["order_w2_W"].append(dc(self.action[:,4]))
            self.state_his["order_w2_M"].append(dc(self.action[:,2]))
            self.state_his["order_W"].append(dc(self.action[:,0]))
            self.state_his["demand_w1"].append(dc(self.demands_w1[:,self.step_num-1]))
            self.state_his["demand_w2"].append(dc(self.demands_w2[:,self.step_num-1]))
            self.state_his["backlog_w1"].append(dc(self.backlog_w1))
            self.state_his["backlog_w2"].append(dc(self.backlog_w2))
            self.state_his["backlog_w1_W"].append(dc(self.backlog_W[:,0]))
            self.state_his["backlog_w2_W"].append(dc(self.backlog_W[:,1]))
            self.state_his["out_W"].append(np.sum(self.order_W, axis = 1))
            self.state_his["out_w1"].append(np.sum(self.order_w1, axis = 1))
            self.state_his["out_w2"].append(np.sum(self.order_w2, axis = 1))
            
            if(done):
                self.draw_pics()

        return np.array(self.state), np.mean(r), done, {"rewards":r}
    
    def state_update(self):

        def forward_order(orders):
            res = orders[:,1:]
            res = np.concatenate([res, np.zeros([res.shape[0], 1])], axis=1)
            return res

        all_goods_received_W = np.sum(self.order_W[:,0])
        all_goods_received_w1 = np.sum(self.order_w1[:,0])
        all_goods_received_w2 = np.sum(self.order_w2[:,0])

        spare_place_W = W_max - np.sum(self.inventory_W)
        spare_place_w1 = w_max - np.sum(self.inventory_w1)
        spare_place_w2 = w_max - np.sum(self.inventory_w2)
        
        if all_goods_received_W == 0:
            lost_rate_M = 1
        else:
            lost_rate_M = np.min([spare_place_W/all_goods_received_W, 1])
        
        if all_goods_received_w1 == 0:
            lost_rate_w1 = 1
        else:
            lost_rate_w1 = np.min([spare_place_w1/all_goods_received_w1, 1])
        
        if all_goods_received_w2 == 0:
            lost_rate_w2 = 1
        else:
            lost_rate_w2 = np.min([spare_place_w2/all_goods_received_w2, 1])
        
        self.inventory_W += np.array(self.order_W[:,0]*lost_rate_M, dtype="int32")
        self.order_W = forward_order(self.order_W)
        self.inventory_w1 += np.array(self.order_w1[:,0]*lost_rate_w1, dtype="int32")
        self.order_w1 = forward_order(self.order_w1)
        self.inventory_w2 += np.array(self.order_w2[:,0]*lost_rate_w2, dtype="int32")
        self.order_w2 = forward_order(self.order_w2)
            
        D = self.action[:,-1] + self.action[:,-2] + self.backlog_W[:,0] + self.backlog_W[:,1]
        D = np.where(D != 0, D, -1)
        rate = self.inventory_W/D
        rate = np.where(rate >= 0, rate, 1)
        rate = np.where(rate <= 1, rate, 1)
        
        W_ship_to_w1 = np.array((self.action[:,-2] + self.backlog_W[:,0])*rate, dtype='int32')
        W_ship_to_w2 = np.array((self.action[:,-1] + self.backlog_W[:,1])*rate, dtype='int32')
        M_ship_to_w1 = self.action[:,-4]
        M_ship_to_w2 = self.action[:,-3]
        M_ship_to_W = self.action[:,-5]
        
        sales_w1 = np.stack([self.inventory_w1, self.demands_w1[:,self.step_num] + self.backlog_w1], axis=1).min(1)
        sales_w2 = np.stack([self.inventory_w2, self.demands_w2[:,self.step_num] + self.backlog_w2], axis=1).min(1)

        self.inventory_w1 -= sales_w1
        self.inventory_w2 -= sales_w2
        self.backlog_w1 = self.backlog_w1 + self.demands_w1[:,self.step_num] - sales_w1
        self.backlog_w2 = self.backlog_w2 + self.demands_w2[:,self.step_num] - sales_w2
        self.backlog_W[:,0] = self.backlog_W[:,0] + self.action[:,-2] - W_ship_to_w1
        self.backlog_W[:,1] = self.backlog_W[:,1] + self.action[:,-1] - W_ship_to_w2
        
        self.inventory_W -= W_ship_to_w1 + W_ship_to_w2
        self.order_W[:,W_M_L-1] += M_ship_to_W
        self.order_w1[:,w1_M_L-1] += M_ship_to_w1
        self.order_w1[:,w1_W_L-1] += W_ship_to_w1
        self.order_w2[:,w2_M_L-1] += M_ship_to_w2
        self.order_w2[:,w2_W_L-1] += W_ship_to_w2
             
        if(self.eval):
            
            cost = self.inventory_W*W_H_C + \
                   self.inventory_w1*w1_H_C + \
                   self.inventory_w2*w2_H_C + \
                   self.backlog_w1*w1_LOST_SALES_C + \
                   self.backlog_w2*w2_LOST_SALES_C + \
                   np.sum(self.backlog_W, axis=1)*W_LOST_SALES_C + \
                   M_ship_to_W*W_ORDER_COST + \
                   (M_ship_to_w1 + M_ship_to_w2)*w_ORDER_COST
            
        else:
            cost = self.normalize(self.normalizer, arr = self.inventory_W, batch = True)*W_H_C + \
                   self.normalize(self.normalizer, arr = self.inventory_w1, batch = True)*w1_H_C + \
                   self.normalize(self.normalizer, arr = self.inventory_w2, batch = True)*w2_H_C + \
                   self.normalize(self.normalizer, arr = self.backlog_w1, batch = True)*w1_LOST_SALES_C + \
                   self.normalize(self.normalizer, arr = self.backlog_w2, batch = True)*w2_LOST_SALES_C + \
                   self.normalize(self.normalizer, arr = np.sum(self.backlog_W, axis=1), batch = True)*W_LOST_SALES_C + \
                   self.normalize(self.normalizer, arr = M_ship_to_W, batch = True)*W_ORDER_COST + \
                   self.normalize(self.normalizer, arr = M_ship_to_w1 + M_ship_to_w2, batch = True)*w_ORDER_COST
            
        rewards = -cost

        self.step_num += 1
        
        return rewards