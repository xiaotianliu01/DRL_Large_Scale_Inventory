import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

REVIEW_PERIOD = [2, 3]
STARTUP_ORDER = 0
GOODS_FEAT_DIM = 1
EPOSIDE_LEN = 75
HIS_DEMAND_LEN = 100
HIS_VLT_LEN = 10

B = 9
H = 1

def interpolate_with_trues(arr, original_length, target_length):
    inter_seq = interp1d([i for i in range(original_length)], arr, kind = 'linear')
    indexes = list(np.linspace(0, original_length-1, target_length - original_length + 1)) + [i for i in range(original_length)]
    indexes.sort()
    inter_res = inter_seq(indexes)
    return inter_res

def extend_seq(seq, target_length):

    start_index = np.random.randint(0, seq.shape[1]-target_length)
    e_seq = seq[:,start_index:start_index+target_length]

    return e_seq.astype(np.int)

class MyEnv(gym.Env):

    def __init__(self, 
                 seed, 
                 demand_data_pth,
                 vlt_data_pth,
                 sku_batch_id = 0, 
                 sku_parallel_batch_num = 1, 
                 eval = False, 
                 draw = None, 
                 pretrain = False):
        
        self.seed(seed)
        self.pretrain = pretrain
        self.sku_batch_id = sku_batch_id
        self.sku_parallel_batch_num = sku_parallel_batch_num
        self.eval = eval
        self.draw = draw
        self.demand_data_pth = demand_data_pth
        self.vlt_data_pth = vlt_data_pth
        self.read_goods_data()

        high_action = np.array([np.finfo(np.float32).max])
        low_action = np.array([0.00])
        self.action_space = spaces.Box(low_action, high_action, dtype=np.float32)
        
        high_obs = np.array([np.finfo(np.float32).max for i in range(6)])

        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.viewer = None
        self.state = None

    def read_goods_data(self):
        df_vlt = pd.read_csv(self.vlt_data_pth)
        lead_times = {}
        for tra in np.array(df_vlt):
            good_index = int(tra[0])
            if good_index in lead_times:
                lead_times[good_index].append(int(tra[1]))
            else:
                lead_times[good_index] = [int(tra[1])]
        
        maximum = -1
        for _, li in lead_times.items():
            maximum = np.max([maximum, np.max(li)])
        self.max_lead = maximum + 1 

        df = pd.read_csv(self.demand_data_pth)
        
        df_data = np.array(df)
        self.goods_demand = []
        self.goods_lead_times = []
        
        if(df_data.shape[0] % self.sku_parallel_batch_num == 0):
            sku_batch_size = df_data.shape[0] // self.sku_parallel_batch_num
        else:
            sku_batch_size = df_data.shape[0] // self.sku_parallel_batch_num + 1
            sample_index = np.random.randint(0, df_data.shape[0], sku_batch_size*self.sku_parallel_batch_num-df_data.shape[0])
            df_data = np.concatenate([df_data, df_data[sample_index][:]], axis = 0)
        
        df_data = df_data[self.sku_batch_id*sku_batch_size:self.sku_batch_id*sku_batch_size+sku_batch_size][:]
        
        for tra in df_data:
            self.goods_demand.append(tra[1:])
            e_lead_times = interpolate_with_trues(np.array(lead_times[int(tra[0])]), len(lead_times[int(tra[0])]), EPOSIDE_LEN+HIS_VLT_LEN+2).astype(np.int)
            self.goods_lead_times.append(np.array(e_lead_times))
        
        self.goods_demand = np.array(self.goods_demand)
        self.goods_lead_times = np.array(self.goods_lead_times)
        self.goods_num = int(self.goods_demand.shape[0])

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def normalize(self, normalizer, arr = [], ele = 0, batch = False):
        
        if(batch == False):
            if(len(arr) == 0):
                if(normalizer[1] > 1e-3):
                    return (ele-normalizer[0])/normalizer[1]
                else:
                    return ele-normalizer[0]
            
            res = []
            for i in arr:
                if(normalizer[1] > 1e-3):
                    res.append((i-normalizer[0])/normalizer[1])
                else:
                    res.append(i-normalizer[0])
            return res
        else:
            if(len(arr.shape) == 1):
                norm_mean = normalizer[0]
                norm_std = np.where(normalizer[1] > 1e-3, normalizer[1], 1)
                return (arr-norm_mean)/norm_std
            else:
                norm_mean = np.expand_dims(normalizer[0], -1)
                norm_std = np.expand_dims(np.where(normalizer[1] > 1e-3, normalizer[1], 1), -1)
                return (arr-norm_mean)/norm_std
    
    def denomalize(self, normalizer, ele):
        return ele*normalizer[1] + normalizer[0]
    
    def get_demand_time_step(self, action_time_step):
        
        res = 0
        for i in range(action_time_step):
            res += self.review_interval[i]
        return res + 1
    
    def draw_pic(self):
        for i in range(self.goods_num):
            times = [ii for ii in range(len(self.state_his["inventory"]))]
            inventory = [ii[i] for ii in self.state_his["inventory"]]
            demand = [ii[i] for ii in self.state_his["demand"]]
            order = [ii[i] for ii in self.state_his["order"]]
            backlog = [ii[i] for ii in self.state_his["backlog"]]
            plt.figure(figsize=(15, 15))
            plt.plot(times, inventory, label = 'I')
            plt.plot(times, demand, label = 'D')
            plt.plot(times, order, label = 'O')
            plt.plot(times, backlog, label = 'B')
            plt.legend()
            plt.savefig(self.draw + str(i) + '.png')
    
    def get_optimal_action(self):
        
        actions = []
        Ss = []
        for i in range(self.goods_num):
            v_m = self.get_demand_time_step(self.step_num) + self.lead_times[i][self.step_num]
            if(self.step_num+1 == len(self.lead_times[i])):
                v_m_1 = self.get_demand_time_step(len(self.lead_times[i])) + 1
            else:
                v_m_1 = self.get_demand_time_step(self.step_num+1) + self.lead_times[i][self.step_num+1] + 1
            S = np.sum(self.demands[i][v_m:v_m + int(B*(v_m_1-v_m)/(B+H))])
            I_L = self.inventory[i] - self.backlog[i]
            for t in range(v_m - self.get_demand_time_step(self.step_num)):
                I_L = I_L - self.demands[i][self.get_demand_time_step(self.step_num) + t]
                I_L += self.order[i][t]
            if(I_L < S):
                a = S - I_L
            else:
                a = 0
            actions.append(a)
            Ss.append(S)
        actions = self.normalize(self.normalizer, np.array(actions), batch = True)/self.review_interval[self.step_num]
        Ss = np.array(Ss)
        return actions, Ss
        
    def reset(self):
        
        self.inventory = np.array([0 for i in range(self.goods_num)])
        self.backlog = np.array([0 for i in range(self.goods_num)])
        self.order = np.array([np.array([STARTUP_ORDER for i in range(self.max_lead)]) for _ in range(self.goods_num)])
        self.review_interval = []
        while True:
            self.review_interval += REVIEW_PERIOD
            if(len(self.review_interval) > EPOSIDE_LEN):
                break
        
        if(self.eval):
            temp_demand = self.goods_demand[:,-(np.sum(self.review_interval) + 20 + HIS_DEMAND_LEN):]
        else:
            temp_demand = self.goods_demand[:,-(np.sum(self.review_interval) + 20 + HIS_DEMAND_LEN):]
        self.demands = temp_demand[:,HIS_DEMAND_LEN:]

        if(self.eval):
            temp_vlt = self.goods_lead_times[:,-(EPOSIDE_LEN+HIS_VLT_LEN+1):]
        else:
            temp_vlt = extend_seq(self.goods_lead_times, EPOSIDE_LEN+HIS_VLT_LEN+1)
        self.lead_times = temp_vlt[:,HIS_VLT_LEN:]
        self.normalizer = [np.array([np.mean(temp_demand[i][:HIS_DEMAND_LEN]) for i in range(self.goods_num)]), np.array([np.std(temp_demand[i][:HIS_DEMAND_LEN]) for i in range(self.goods_num)])]
        self.lead_times_normalizer = [np.array([np.mean(temp_vlt[i][:HIS_VLT_LEN]) for i in range(self.goods_num)]), np.array([np.std(temp_vlt[i][:HIS_VLT_LEN]) for i in range(self.goods_num)])]
        
        self.step_num = 0
        
        state_temp_1 = np.stack([self.review_interval[0]*np.ones([self.goods_num]), self.normalize(self.lead_times_normalizer, self.lead_times[:,0], batch = True)], axis=1)

        sum_orders = np.sum(self.order, axis=1) 
        
        norm_inv = self.normalize([self.normalizer[0]*self.review_interval[0],self.normalizer[1]*(self.review_interval[0])**0.5], self.inventory, batch=True)
        norm_bac = self.normalize([self.normalizer[0]*self.review_interval[0],self.normalizer[1]*(self.review_interval[0])**0.5], self.backlog, batch=True)
        norm_ord = self.normalize([self.normalizer[0]*self.review_interval[0],self.normalizer[1]*(self.review_interval[0])**0.5], sum_orders, batch=True)
        norm_demand = self.normalize(self.normalizer, self.demands[:,0], batch=True)
        state_temp_2 =  np.stack([norm_demand, norm_inv, norm_bac, norm_ord], axis=1)
        self.state = np.concatenate([state_temp_1, state_temp_2], axis=1)
        
        if(self.draw):
            self.state_his = {"inventory":[], "order":[], "demand":[], "backlog":[]}

        if(self.pretrain):
            optimal_actions, S = self.get_optimal_action()
            self.last_optimal = optimal_actions
            self.last_optimal_S = S

        return np.array(self.state)
    
    def step(self, action, true_action = False):
        self.action = action
        if(true_action == False):
            de_actions = np.zeros_like(self.action)
            for i in range(self.action.shape[0]):
                for j in range(self.action.shape[1]):
                    de_actions[i][j] = int(np.max([self.denomalize([self.normalizer[0][i], self.normalizer[1][i]], self.action[i][j])*self.review_interval[self.step_num], 0]))                    
            self.action = de_actions
        
        r = self.state_update()

        state_temp_1 = np.stack([self.review_interval[self.step_num]*np.ones([self.goods_num]), self.normalize(self.lead_times_normalizer, self.lead_times[:,self.step_num-1], batch=True)], axis=1)
        sum_orders = np.sum(self.order, axis=1)

        norm_inv = self.normalize([self.normalizer[0]*self.review_interval[self.step_num], self.normalizer[1]*(self.review_interval[self.step_num])**0.5], self.inventory, batch=True)
        norm_bac = self.normalize([self.normalizer[0]*self.review_interval[self.step_num], self.normalizer[1]*(self.review_interval[self.step_num])**0.5], self.backlog, batch=True)
        norm_ord = self.normalize([self.normalizer[0]*self.review_interval[self.step_num], self.normalizer[1]*(self.review_interval[self.step_num])**0.5], sum_orders, batch=True)
        norm_demand = self.normalize(self.normalizer, self.demands[:,self.get_demand_time_step(self.step_num)-1], batch=True)
        state_temp_2 =  np.stack([norm_demand, norm_inv, norm_bac, norm_ord], axis=1)
        self.state = np.concatenate([state_temp_1, state_temp_2], axis=1)
    
        if(self.step_num == EPOSIDE_LEN):
            done = True
        else:
            done = False
        
        if(done and self.draw):
            self.draw_pic()
        
        if(self.pretrain):
            import copy
            infos = {"rewards":r, "optimal_actions": copy.deepcopy(self.last_optimal), "optimal_S": self.last_optimal_S}
            self.last_optimal, self.last_optimal_S = self.get_optimal_action()
            return np.array(self.state), np.mean(r), done, infos
        
        return np.array(self.state), np.mean(r), done, {"rewards":r}

    def state_update(self):
        
        r = np.array([0 for i in range(self.goods_num)], dtype = 'float64')
        demand_time_step = self.get_demand_time_step(self.step_num)
        for j in range(self.review_interval[self.step_num]): 
            demand = self.demands[:,demand_time_step + j] + self.backlog
            sellable = self.inventory + self.order[:,0] 
            sales = np.stack([demand, sellable], axis=1).min(1)
            self.inventory = sellable - sales 
            self.backlog = np.where(demand-sales > 0, demand-sales, 0)
            self.order = self.order[:,1:] 
            self.order = np.concatenate([self.order, np.zeros([self.goods_num, 1])], axis=1)
            if(j == 0):
                L = self.lead_times[:,self.step_num]
                O = np.eye(self.order.shape[1])[L]*self.action
                self.order += O
            
            if(self.draw):
                if(j == 0):
                    self.state_his['order'].append(self.action)
                else:
                    self.state_his['order'].append(np.zeros([self.goods_num]))
                self.state_his['inventory'].append(self.inventory)
                self.state_his['backlog'].append(self.backlog)
                self.state_his['demand'].append(self.demands[:,demand_time_step + j])            
            
            if(self.eval):
                r += -(B*self.backlog + H*self.inventory)
            else:
                r += -(B*self.normalize(self.normalizer, arr = self.backlog, batch = True) + H*self.normalize(self.normalizer, arr = self.inventory, batch = True))
                r = r/(B+H)

        if(self.eval):
            r_ = r
        else:
            r_ = [i/self.review_interval[self.step_num] for i in r]
        self.step_num += 1
        return r_
        