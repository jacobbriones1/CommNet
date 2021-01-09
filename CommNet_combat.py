import gym
import ma_gym
import time
import torch
from torch import nn
from Modules import *
from utils import *

env = gym.make('Combat-v0')
actions = env.get_action_meanings()[0]
num_actions = len(actions)
num2ac = dict()
ac2num = dict()
for i in range(num_actions):
    num2ac[i]=actions[i]
    ac2num[actions[i]]=i

num_agents = env.n_agents
output_size = 50
comm_steps = 5
CN = CommNetMLP(num_agents,150, output_size, num_actions, comm_steps)
t=0
ep_rewards = []
num_episodes = 50
for _ in range(num_episodes):
    episode_reward = 0
    done_n = [False for _ in range(num_agents)]
    s = env.reset()
    env.render()
    while all(done_n)==False:
        a = sample_distributions(CN(s))
        s, reward, done_n, h= env.step(a)
        env.render()
        episode_reward += sum(reward)
    env.close()
    print('Episode Reward:', episode_reward)
    ep_rewards.append(episode_reward)
        
        
    
