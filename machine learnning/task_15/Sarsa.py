import sys
import time
import gym
import numpy as np


class SarsaAgent():
    def __init__(self, dim_state, dim_action):
        self.dim_state = dim_state
        self.dim_action = dim_action
        # Q评估Q+A的表
        self.Q_TABLE = np.zeros((dim_state, dim_action))  # [dim_state dim_action]

    def choose_action(self, state, explode=True):
        if explode and np.random.uniform() > 0.95:  # >0.9则选择
            action = np.random.choice(self.dim_action)  # 随机选择一个方法
        else:
            Q_LIST = self.Q_TABLE[state, :]  # 这个state下所有的action的值
            max_Q = np.max(Q_LIST)  # 最大的action
            action_list = np.where(Q_LIST == max_Q)[0]  # 最大值可能有多个 返回索引 []
            action = np.random.choice(action_list)  # 最大值随机选取
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        predict_Q = self.Q_TABLE[state, action]  # Q_table表找到对应Q的评估值
        if done:
            target_Q = reward
        else:
            # Q <- Q + a*[(R + y*next_Q) - Q] next_Q:采取实际action 产生
            target_Q = reward + 0.99 * self.Q_TABLE[next_state, next_action]
        self.Q_TABLE[state, action] = self.Q_TABLE[state, action] + 0.001 * (target_Q - predict_Q)


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    # env = gym.make("CliffWalking-v0",render_mode="human")

    dim_state = env.observation_space.n  # 48
    dim_action = env.action_space.n  # 4

    print("dim_state", dim_state)
    print("dim_action", dim_action)
    agent = SarsaAgent(dim_state, dim_action)

    for epoch in range(5000):
        state, info = env.reset()  # 开始一局游戏

        total_rewards, total_steps = 0, 0
        action = agent.choose_action(state)
        while True:
            # env.render()
            next_state, reward, terminated, truncated, info = env.step(action)  # 采取该行为获取下一个state 及分数
            # print(next_state, reward, terminated, truncated, info)
            next_action = agent.choose_action(next_state)  # 行为 概率
            if next_state == 47: reward = 10
            total_rewards = total_rewards + reward
            agent.learn(state, action, total_rewards, next_state, next_action, terminated or truncated)
            action = next_action
            state = next_state
            if terminated or truncated:
                break

    # EVAL
    env = gym.make("CliffWalking-v0", render_mode="human")
    state, info = env.reset()  # 开始一局游戏
    env.render()

    action = agent.choose_action(state)
    while True:

        next_state, reward, terminated, truncated, info = env.step(action)  # 采取该行为获取下一个state 及分数
        print(reward)
        if state == 47: reward = 100
        next_action = agent.choose_action(next_state, False)  # 行为 概率
        action = next_action
        state = next_state
        if terminated: break
        time.sleep(0.5)
    print(f"Epoch:{epoch}|Total_steps:{total_steps}|Total_rewards:{total_rewards}")
    print(agent.Q_TABLE)
    print(agent.Q_TABLE[24])
    time.sleep(1000)
