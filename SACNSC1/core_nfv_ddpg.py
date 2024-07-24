from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
# from enviroment import Env_cellular as env
import matplotlib.pyplot as plt
import pickle
import random
from scipy.stats import rayleigh

#%%####################  larning parameters  ####################

Pn = 1
K = 2
MAX_EPISODES = 4000
MAX_EP_STEPS = 200
# LR_A = 0.000001    # learning rate for actor
# LR_C = 0.000001    # learning rate for critic
GAMMA = 0.6   # reward discount
TAU = 0.5    # soft replacement
MEMORY_CAPACITY = 100000

###############################  DDPG  ####################################
class Agent(object):
    def __init__(self,LR_A,LR_C, a_dim ,s_dim,BATCH_SIZE,):
        # self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.memory_trans= deque(maxlen=100000)  
        self.BATCH_SIZE=BATCH_SIZE
        self.a_dim=a_dim
        self.a_bound=1
        self.a_dim, self.s_dim = a_dim, s_dim, 
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            print(q_target)
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())
        
        

    def choose_action(self, s):
        self.s=s
        self.s=np.reshape(s,[1,self.s_dim])
        ac=self.sess.run(self.a, {self.S: self.s })[0]
        ac=np.clip(ac,-1,1)
        ac=(ac+1)/2
        return ac

    def replay(self):
        if len(self.memory_trans) < self.BATCH_SIZE:
            self.BATCH_SIZE= self.pointer
        minibatch = random.sample(self.memory_trans, self.BATCH_SIZE)
        for state, action, reward,state_ in minibatch:
            self.bs = state
            
            self.bs_ = state_
            self.ba = action
            self.br= reward 
           
        # bt = self.memory_trans[indices, :] ## اینجا
        # bs = bt[:, :self.s_dim]
        # ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        # br = bt[:, -self.s_dim - 1: -self.s_dim]
        # bs_ = bt[:, -self.s_dim:]
        self.bs=np.reshape(self.bs,[1,self.s_dim])
        self.bs_=np.reshape(self.bs_,[1,self.s_dim])

        self.sess.run(self.atrain, {self.S: self.bs})
        
        self.ba=np.reshape(self.ba,[1,self.a_dim])
        self.sess.run(self.ctrain, {self.S: self.bs, self.a: self.ba, self.R: self.br, self.S_: self.bs_})
        return   
        
    def memorize(self, s, a, r, s_):
        r = np.reshape(r,(1,1))
        a = np.reshape(a,[self.a_dim])
        #print(f"state is {s}, action is {a}, reward is {r}, next state is {s_}")
        self.memory_trans.append((s, a, r, s_)) ## اینجا
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        # self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 1000, activation=tf.nn.relu, name='l1', trainable=trainable)
            a2 = tf.layers.dense(net, 500, activation=tf.nn.tanh, name='l2', trainable=trainable)
            a3 = tf.layers.dense(a2, 250, activation=tf.nn.tanh, name='l3', trainable=trainable)
            a4 = tf.layers.dense(a3, 25, activation=tf.nn.tanh, name='l4', trainable=trainable)
            a = tf.layers.dense(a4, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net, 500, activation=tf.nn.relu, name='lx2', trainable=trainable)
            net3 = tf.layers.dense(net2, 250, activation=tf.nn.relu, name='lx3', trainable=trainable)
            net4 = tf.layers.dense(net3, 25, activation=tf.nn.relu, name='lx4', trainable=trainable)
            #not sure about this part
            q=tf.layers.dense(net4, 1, trainable=trainable)
            return q  # Q(s,a)

