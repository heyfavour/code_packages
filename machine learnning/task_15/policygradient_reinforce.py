# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gym

EPISODE_PER_BATCH = 5  # 每蒐集 5 個 episodes 更新一次 agent
NUM_BATCH = 400  # 總共更新 400 次

#env = gym.make('LunarLander-v2')
env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]  # 4 state的维度
# to confirm the shape
# 确定action的shape
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):  # Actor
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 16),
            nn.Tanh(),
            nn.Linear(16, N_ACTIONS),
            nn.Softmax(dim=-1)  # dim=2 最后一维 行
        )

    def forward(self, state):
        output = self.model(state)
        return output


class Policy():
    def __init__(self, network):
        self.network = network
        self.optimizer = optim.AdamW(self.network.parameters(), lr=0.001)

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))  # 根据state出行为分布[p1 p2 p3 p4]
        action_dist = Categorical(action_prob)  # 创建以参数probs为标准的类别分布，样本是来自“0，...，K-1”的整数
        action = action_dist.sample()  # 采样 aciont 1 2 3 4 按概率sample
        log_prob = action_dist.log_prob(action)  # 获取log动作概率  lnp1 lnp2 lnp3 lnp4
        return action.item(), log_prob

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()  # log_probs lnp*reward = torch [0.1]

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        action = np.argmax(self.network(torch.FloatTensor(state)).detach().numpy(),axis=0).item()
        return action


if __name__ == '__main__':

    network = Net()  # 根据场景分动作的概率
    agent = Policy(network)  #
    agent.network.train()  # 訓練前，先確保 network 處在 training 模式
    avg_total_rewards, avg_final_rewards = [], []
    for batch in range(NUM_BATCH):
        log_probs, rewards, total_rewards, final_rewards = [], [], [], []
        for episode in range(10):  # EPISODE_PER_BATCH  5 個 episodes 更新一次 nn
            state = env.reset()  # 开始一局游戏
            total_reward, total_step = 0, 0
            while True:
                # env.render()
                action, log_prob = agent.sample(state)  # 行为 a 概率 LN P(a)
                next_state, reward, done, _ = env.step(action)  # 采取该行为获取下一个state 及分数
                log_probs.append(log_prob)  # 行为概率
                state = next_state
                total_reward = total_reward + reward  # 总分
                total_step = total_step + 1  # 步长
                if done:
                    final_rewards.append(reward)  # 总分
                    total_rewards.append(total_reward)  # 总步长
                    # 設定同一個 episode 每個 action 的 reward 都是 total reward
                    # append([total_reward total_reward total_reward ...]) len=total_step
                    rewards.append(np.full(total_step, total_reward))
                    break
        # 紀錄訓練過程
        rewards = np.concatenate(rewards, axis=0)  # 5次游戏的拼接数组 [r r r ] len=step
        avg_total_reward = sum(total_rewards) / len(total_rewards)  # 每一步的平均分
        avg_final_reward = sum(final_rewards) / len(final_rewards)  # 5次游戏最后一步的平均分
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # 將 reward 标准化 (x-mean/std)
        agent.learn(torch.stack(log_probs), torch.from_numpy(rewards)) #[p p p p] [r r r r]
        #print(f"[{batch:0=3d}/{NUM_BATCH}] Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")
        print(f"[{batch:0=3d}/{NUM_BATCH}] total_rewards: {total_rewards},")

    agent.network.eval()  # 測試前先將 network 切換為 evaluation 模式

    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.predict(state)
        state, reward, done, _ = env.step(action)
        total_reward = total_reward + reward
    print(total_reward)
    #训练效果比DQN慢
