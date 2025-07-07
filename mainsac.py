

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:34:06 2022

@author: Maryam & Sajedeh
"""
# importing the time module.


import time
import numpy as np
import os
from Env import Env
from sac_agent import SACAgent as Agent
#import matplotlib.pyplot as plt
import scipy.io 

np.random.seed(1376)


#%%
n_vehicle = 40
n_paths = 24
n_subchannel = 10
env = Env(n_vehicle, n_subchannel, n_paths)
#%% _____ Learning params________
n_episode = 1000
Tf = 50000
n_episode_test = 20                    # test episodes
batch_size = 32
alpha=0.0000001
beta=0.0000001
gamma=0.9
tau=0.01
max_size=1000000 
fc1_dims=400
fc2_dims=300
# actor and critic hidden layers
C_fc1_dims = 1024
C_fc2_dims = 512
C_fc3_dims = 256

A_fc1_dims = 1024
A_fc2_dims = 512
V_fc1_dims = 256 
V_fc2_dims = 256
memory_size = 1000000
#%%
IS_TRAIN = 1
IS_TEST = 1 - IS_TRAIN
label = 'SAC'
#label2 = 'baseline'
label2= 'agdrp'
current_dir = os.path.dirname(os.path.realpath(__file__))
reward_path      = os.path.join (current_dir, "model/" +label +label2 +'/reward.mat')
TTlink_path      = os.path.join (current_dir, "model/" +label +label2 +'/TT_link.mat')
TTvehicle_path   = os.path.join (current_dir, "model/" +label +label2 +'/TT_vehicle.mat')
Agevehicle_path  = os.path.join (current_dir, "model/" +label +label2 +'/Age_vehicle.mat')
Flagvehicle_path = os.path.join (current_dir, "model/" +label +label2 +'/Flage_vehicle.mat')
start = time.time()

n_input = (8 * n_vehicle) + n_paths# vehicle location(2V), flag_vehicle, age, selected point, capacity
n_output =   n_vehicle + (n_vehicle * n_subchannel) # selected point, channel assignment

#%% Learning to choose winners 
# agent = Agent(alpha, beta, n_input, tau, n_output, gamma,
#               max_size, fc1_dims, fc2_dims, batch_size)

agent = Agent(alpha, beta, n_input, tau, n_output, gamma, memory_size, C_fc1_dims, C_fc2_dims, C_fc3_dims,
                  A_fc1_dims, A_fc2_dims, V_fc1_dims, V_fc2_dims, batch_size, 1) 

score_history = []
TT_link_e = []
TT_vehicle_e = []
AGE_vehicle_e = []
reward_e = []
FLAG_vehicle_e= []
#bb=1
#decay=.9999

for episode in range(n_episode):
    if episode !=0:
        agent.load_models()
    obs = env.start_point()
    done = False
    score = 0
    #agent.noise.reset()
    r = []
    TT_link = []
    TT_vehicle = []
    AGE_vehicle = []
    FLAG_vehicle = []
    #while not done:
    for step in range(1000):
        #------------------
        #bb = bb*decay
        #obs=obs/(1+max(obs))
        #------------------
        action = agent.choose_action(obs)
        obs_, reward, tt_link, tt_vehicle, age_vehicle, flag_vehicle = env.step(action, obs)
        #------------------
        #obs_=obs_/(1+max(obs_))
        #------------------
        agent.remember(obs, action, reward, obs_, done)
        agent.learn()
        score += reward
        print ('episode', episode)
        print('step:', step)
        print(reward)

        print('-----')
        obs = obs_
        r.append(reward)
        TT_link.append(tt_link)
        TT_vehicle.append(tt_vehicle)
        AGE_vehicle.append(age_vehicle)
        FLAG_vehicle.append(flag_vehicle)
        score_history.append(score)
        avg_score = np.mean(score_history[-1000:])
     
    reward_e.append(np.mean(r))
    TT_link_e.append(np.mean(tt_link))
    TT_vehicle_e.append(np.mean(tt_vehicle))
    AGE_vehicle_e.append(np.mean(age_vehicle))
    FLAG_vehicle_e.append(np.mean(FLAG_vehicle))
    agent.save_models()
    scipy.io.savemat(reward_path,      {'reward':        np.array (reward_e)})
    scipy.io.savemat(TTlink_path,      {'TT_link':       np.array (TT_link_e)})
    scipy.io.savemat(TTvehicle_path,   {'TT_vehicle':    np.array (TT_vehicle_e)})
    scipy.io.savemat(Agevehicle_path,  {'Age_vehicle':   np.array (AGE_vehicle_e)})
    scipy.io.savemat(Flagvehicle_path, {'Flage_vehicle': np.array (FLAG_vehicle_e)})
