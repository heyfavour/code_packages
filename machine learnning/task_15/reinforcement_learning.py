# -*- coding: utf-8 -*-
"""reinforcement_learning

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FybeGBvuJ79pZankhvcfGhuEnTTfkkO_
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !apt update
# !apt install python-opengl xvfb -y
# !pip install gym[box2d] pyvirtualdisplay

# Commented out IPython magic to ensure Python compatibility.
from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
# %matplotlib inline
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import gym
env = gym.make('LunarLander-v2')

# initial_state = env.reset()#env.observation_space 得到一个初始的env数据 [8]
# img = plt.imshow(env.render(mode='rgb_array'))
# plt.show()

# done = False
# while not done:
#   random_action = env.action_space.sample()#env.action_space=[4]
#   observation, reward, done, info = env.step(random_action)
#   print(reward, done)
  #img = plt.imshow(env.render(mode='rgb_array'))
  #display.display(plt.gcf())
  #display.clear_output(wait=True)

class PolicyGradientNetwork(nn.Module):#Actor
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Linear(8, 16),
        nn.Tanh(),
        nn.Linear(16, 16),
        nn.Tanh(),
        nn.Linear(16, 4),
        nn.Softmax(dim=-1)#dim=2 最后一维 行
    )

  def forward(self, state):
    output = self.model(state)  
    return output
#from torchsummary import summary
#model = PolicyGradientNetwork()
#summary(model,(8,))

class PolicyGradientAgent():
  def __init__(self, network):
    self.network = network
    self.optimizer = optim.AdamW(self.network.parameters(), lr=0.001)

  def learn(self, log_probs, rewards):
    loss = (-log_probs * rewards).sum()#log_probs lnp*reward = torch [0.1]

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

  def sample(self, state):
    action_prob = self.network(torch.FloatTensor(state))#根据state出行为分布[p1 p2 p3 p4]
    action_dist = Categorical(action_prob)#创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数
    action = action_dist.sample()#采样 aciont 1 2 3 4
    log_prob = action_dist.log_prob(action)#获取动作的概率的对数 ln
    return action.item(), log_prob

network = PolicyGradientNetwork()#根据场景分动作的概率
agent = PolicyGradientAgent(network)#

agent.network.train()  # 訓練前，先確保 network 處在 training 模式
EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 1        # 總共更新 400 次

avg_total_rewards, avg_final_rewards = [], []

for batch in range(NUM_BATCH):
  log_probs, rewards = [], []
  total_rewards, final_rewards = [], []

  for episode in range(5):#EPISODE_PER_BATCH  5 個 episodes 更新一次 agent
    state = env.reset()#开始一局游戏
    total_reward, total_step = 0, 0
    while True:
      action, log_prob = agent.sample(state)#行为 概率
      next_state, reward, done, _ = env.step(action)#采取该行为获取下一个state 及分数
      log_probs.append(log_prob)#行为概率
      state = next_state
      total_reward = total_reward + reward#总分
      total_step = total_step + 1 #步长
      if done:
        final_rewards.append(reward)#总分
        total_rewards.append(total_reward)#总步长
        rewards.append(np.full(total_step, total_reward))#[r,r,r, ..,r]  # 設定同一個 episode 每個 action 的 reward 都是 total reward
        break

  # 紀錄訓練過程
  avg_total_reward = sum(total_rewards) / len(total_rewards)#每一步的平均分
  avg_final_reward = sum(final_rewards) / len(final_rewards)#5次游戏最后一步的平均分
  avg_total_rewards.append(avg_total_reward)#每一步的平均分
  avg_final_rewards.append(avg_final_reward)#5次游戏最后一步的平均分
  print(f"[{batch:0=3d}/{NUM_BATCH}] Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

  # 更新網路
  rewards = np.concatenate(rewards, axis=0)#5次游戏的拼接数组 [r r r ]len=step
  rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 正規標準化
  agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))

plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()

from google.colab import drive

drive.mount('/content/drive')
path = "drive/MyDrive/app/"

import os
torch.save(network.state_dict(), os.path.join(path, f'PolicyGradientNetwork.pth'))
#torch.save(agent.state_dict(), os.path.join(path, f'PolicyGradientAgent.pth'))

agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式
state = env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
total_reward = 0

done = False
while not done:
  action, _ = agent.sample(state)
  state, reward, done, _ = env.step(action)

  total_reward += reward

  img.set_data(env.render(mode='rgb_array'))
  display.display(plt.gcf())
  display.clear_output(wait=True)