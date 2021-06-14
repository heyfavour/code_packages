import gym
import time
import numpy as np

from gridworld import CliffWalkingWapper

EPSILON = 0.9
GAMMA = 0.9
LEARNING_RATE = 0.1


class QLearningAgent():
    def __init__(self, dim_state, dim_action):
        self.dim_state = dim_state
        self.dim_action = dim_action
        # Q评估Q+A的表
        self.Q_TABLE = np.zeros((dim_state, dim_action))  # [dim_state dim_action]

    def choose_action(self, state,explode=True):
        if np.random.uniform() > EPSILON and explode:  # >0.9则选择
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
            # Q <- Q + a*[(R + y*next_Q) - Q] Q_TABLE预测 max_action(Q(s',a))
            target_Q = reward + GAMMA * np.max(self.Q_TABLE[next_state,:])
        self.Q_TABLE[state, action] = self.Q_TABLE[state, action] + LEARNING_RATE * (target_Q - predict_Q)


if __name__ == '__main__':
    env = gym.make("CliffWalking-v0")
    env = CliffWalkingWapper(env)
    dim_state = env.observation_space.n  # 48
    dim_action = env.action_space.n  # 4
    agent = QLearningAgent(dim_state, dim_action)

    for epoch in range(500):
        state = env.reset()  # 开始一局游戏

        total_rewards, total_steps = 0, 0
        action = agent.choose_action(state)
        while True:
            #env.render()
            next_state, reward, done, _ = env.step(action)  # 采取该行为获取下一个state 及分数
            next_action = agent.choose_action(next_state)  # 行为 概率
            agent.learn(state, action, reward, next_state, next_action, done)
            action = next_action
            state = next_state
            total_steps = total_steps + 1
            total_rewards = total_rewards + reward
            if done:break
            print(f"Epoch:{epoch}|Total_steps:{total_steps}|Total_rewards:{total_rewards}")
    #EVAL
    state = env.reset()  # 开始一局游戏

    action = agent.choose_action(state)
    while True:
        env.render()
        next_state, reward, done, _ = env.step(action)  # 采取该行为获取下一个state 及分数
        next_action = agent.choose_action(next_state,False)  # 行为 概率
        action = next_action
        state = next_state
        if done: break
        time.sleep(0.5)
    print(f"Epoch:{epoch}|Total_steps:{total_steps}|Total_rewards:{total_rewards}")
    print(agent.Q_TABLE)

