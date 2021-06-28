"""
RAINBOW 7种混合
1.DDQN
2.duelingDQN 加入后收敛效果变慢
"""
import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 定义参数
BATCH_SIZE = 128  # 每一批的训练量
LR = 0.001  # 学习率
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target的更新频率
MEMORY_CAPACITY = 2000  # memery较小时，训练较快，但是后期效果没记忆体大好，建议动态记忆体，前期记忆体小，后期记忆体大

env = gym.make('CartPole-v0')
env = env.unwrapped  # 还原env的原始设置，env外包了一层防作弊层
N_ACTIONS = env.action_space.n  # 2 2个动作
N_STATES = env.observation_space.shape[0]  # 4 state的维度
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

ATOMS_NUM = 51
V_MIN = -10
V_MAX = 10
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOMS_NUM)

class Memory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=MEMORY_CAPACITY):
        self.memory = np.zeros(memory_size, dtype=object)
        self.memory_size = memory_size
        self.memory_counter = 0

    def add(self, transition):  # data = (state, action, reward, next_state, done) 4 1 1 4 1
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self, batch_size=BATCH_SIZE):
        sample_index = np.random.choice(self.memory_size, batch_size)  # 2000中取一个batch index [batch_size]
        # [(s a r s) ] -> [(s) (a) (r) (s)]
        sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()
        sample_state = torch.FloatTensor(np.vstack(sample_memory[0]))
        sample_action = torch.LongTensor(list(sample_memory[1])).view(-1, 1)
        sample_reward = torch.FloatTensor(list(sample_memory[2])).view(-1, 1)
        sample_next_state = torch.FloatTensor(np.vstack(sample_memory[3]))
        sample_done = torch.FloatTensor(sample_memory[4].astype(int)).view(-1, 1)
        return sample_state, sample_action, sample_reward, sample_next_state, sample_done


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear":
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 50),
            nn.ReLU(),
            nn.Linear(50, 50),  # 2层隐藏层以后效果更好 一层时，到200d多epoch才收敛
            nn.ReLU(),
            nn.Linear(50, N_ACTIONS),
            # nn.Softmax(dim=-1), 如果加了很难收敛
        )
        # self.apply(weights_init)  # initialization

    def forward(self, input):
        output = self.model(input)
        return output


class Agent():
    def __init__(self):
        self.epsilon = 0.5
        self.epsilon_max = 0.95
        self.epsilon_step = 5000
        self.epsilon_decay = (self.epsilon_max - self.epsilon) / self.epsilon_step

        self.memory = Memory(MEMORY_CAPACITY)
        self.eval_net, self.target_net = Net(), Net()
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=LR)

        self.learn_step_counter = 0
        self.update_target_model()
        self.loss_func = nn.MSELoss()

    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:  # 贪婪策略
            action = self.predict_action(state)
        else:
            action = np.random.randint(0, N_ACTIONS)  # 随机产生一个action
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.eval_net(state)  # [1 2]
            action = actions_value.max(1)[1].item()
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def learn(self):
        if self.epsilon < self.epsilon_max: self.epsilon = self.epsilon + self.epsilon_decay
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: self.update_target_model()
        self.learn_step_counter = self.learn_step_counter + 1

        sample_state, sample_action, sample_reward, sample_next_state, sample_done = self.memory.sample(BATCH_SIZE)

        q_eval = self.eval_net(sample_state).gather(1, sample_action)
        q_next = self.target_net(sample_next_state).detach()
        q_target = sample_reward + (1 - sample_done) *GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1) TD MC?

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# modify the reward 如果不重定义分数，相当难收敛
def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward

def same_seeds(seed):
    # 每次运行网络的时候相同输入的输出是固定的
    torch.manual_seed(seed)  # 初始化种子保持一致
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.初始化种子保持一致
    random.seed(seed)  # Python random module. 初始化种子保持一致
    # 内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
    # 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    torch.backends.cudnn.benchmark = False
    # 将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    same_seeds(0)
    agent = Agent()
    for epoch in range(400):
        state = env.reset()
        epoch_rewards = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = modify_reward(next_state)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

            epoch_rewards = epoch_rewards + reward
            if agent.memory.memory_counter >= MEMORY_CAPACITY:
                agent.learn()
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
            if done:break
            state = next_state
