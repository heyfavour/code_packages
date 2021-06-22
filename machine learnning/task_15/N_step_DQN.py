"""
忽然感觉把memery拆出来真的香
"""
import torch
import torch.nn as nn
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.001  # learning rate
EPSILON = 0.9  # greedy policy
GAMMA = 0.9  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 1000
# env = gym.make('Pendulum-v0')
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  #
N_STATES = env.observation_space.shape[0]


class Memory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=MEMORY_CAPACITY, n_multi_step=2):
        self.memory = np.zeros(memory_size, dtype=object)
        self.memory_size = memory_size
        self.memory_counter = 0
        self.n_multi_step = n_multi_step

    def add(self, transition):  # data = (state, action, reward, next_state, done) 4 1 1 4 1
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self, batch_size=BATCH_SIZE):
        if self.n_multi_step == 1:
            sample_index = np.random.choice(self.memory_size, batch_size)  # 2000中取一个batch index [batch_size]
            # [(s a r s) ] -> [(s) (a) (r) (s)]
            sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()
            sample_state = torch.FloatTensor(np.vstack(sample_memory[0]))
            sample_action = torch.LongTensor(list(sample_memory[1])).view(-1, 1)
            sample_reward = torch.FloatTensor(list(sample_memory[2])).view(-1, 1)
            sample_next_state = torch.FloatTensor(np.vstack(sample_memory[3]))
            sample_done = torch.FloatTensor(sample_memory[4].astype(int)).view(-1, 1)
        else:
            # 2000中取一个batch index [batch_size--self.n_multi_step] 防止数据不够、
            sample_index = np.random.choice(self.memory_size - self.n_multi_step, batch_size)
            sample_index = (self.memory_counter + sample_index) % self.memory_size
            # [(s a r s) ] -> [(s) (a) (r) (s)]
            sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()
            sample_state = torch.FloatTensor(np.vstack(sample_memory[0]))
            sample_action = torch.LongTensor(list(sample_memory[1])).view(-1, 1)
            # 处理N步step的state+reward
            sample_reward = np.zeros(batch_size, dtype=float)
            sample_next_state = np.zeros(batch_size, dtype=object)
            sample_done = np.zeros(batch_size, dtype=float)
            for index, sample_index in enumerate(sample_index):
                reward = [self.memory[(sample_index + i) % self.memory_size][2] for i in range(self.n_multi_step)]
                next_state = [self.memory[(sample_index + i) % self.memory_size][3] for i in range(self.n_multi_step)]
                done = np.array(
                    [self.memory[(sample_index + i) % self.memory_size][4] for i in range(self.n_multi_step)]
                )
                if sample_done.max() == True:
                    max_index = done.argmax(0) + 1
                    reward = reward[0:max_index]  #
                    next_state = next_state[0:max_index]
                    done = done[0:max_index]
                sample_reward[index] = sum(
                    [(GAMMA ** step) * reward for step, reward in enumerate(reward)]
                )  # n_step rewards
                sample_next_state[index] = next_state[-1]
                sample_done[index] = done[-1]
        return sample_state, sample_action, sample_reward, sample_next_state, sample_done


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear": m.weight.data.normal_(0.0, 0.1)
    if classname == "Linear": m.bias.data.fill_(0)


class NET(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(N_STATES, 20),
            nn.ReLU(),
            nn.Linear(20, N_ACTIONS),
        )
        self.apply(weights_init)  # initialization

    def forward(self, input):
        output = self.model(input)
        return output


class NStepDQN():
    def __init__(self):
        self.epsilon = 0.5
        self.epsilon_max = 0.99
        self.epsilon_step = 5000
        self.epsilon_decay = (self.epsilon_max - self.epsilon) / self.epsilon_step

        self.memory = Memory(MEMORY_CAPACITY)
        self.eval_net, self.target_net = NET(), NET()
        self.optimizer = torch.optim.AdamW(self.eval_net.parameters(), lr=LR)

        self.learn_step_counter = 0
        self.update_target_model()
        self.loss_func = nn.MSELoss()

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    # get action from model using epsilon-greedy policy
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

    # save sample (error,<s,a,r,s'>) to the replay memory
    def store_transition(self, state, action, reward, next_state, done):
        # transition = np.hstack((state, [action, reward], next_state))
        self.memory.add((state, action, reward, next_state, done))

    # pick samples from prioritized replay memory (with batch_size)
    def learn(self):
        # step 1 更新epsilon 慢慢提高策略的选取机率
        if self.epsilon < self.epsilon_max: self.epsilon = self.epsilon + self.epsilon_decay
        # step 2  每N步更新一次target_net
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0: self.update_target_model()
        self.learn_step_counter = self.learn_step_counter + 1

        sample_state, sample_action, sample_reward, sample_next_state, sample_done = self.memory.sample(BATCH_SIZE)
        sample_state = torch.FloatTensor(sample_state)
        sample_action = torch.LongTensor(sample_action).view(-1, 1)
        sample_reward = torch.FloatTensor(sample_reward).view(-1, 1)
        sample_next_state = torch.FloatTensor(list(sample_next_state))
        sample_done = torch.FloatTensor(sample_done).view(-1, 1)
        q_eval = self.eval_net(sample_state).gather(1, sample_action)  # 去对应的acion的实际output
        q_next = self.target_net(sample_next_state).detach()
        # target = rewards + (1 - dones) * GAMMA * next_pred.max(1)[0]
        q_target = sample_reward + (1 - sample_done) * (GAMMA ** self.memory.n_multi_step) * q_next.max(1)[0].view(BATCH_SIZE, 1)

        self.optimizer.zero_grad()
        loss = self.loss_func(q_eval, q_target)
        loss.backward()
        self.optimizer.step()



def modify_reward(state):
    x, x_dot, theta, theta_dot = state  # (位置x，x加速度, 偏移角度theta, 角加速度)
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward


if __name__ == "__main__":
    agent = NStepDQN()
    for epoch in range(400):
        state = env.reset()
        epoch_rewards = 0
        while True:
            # env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            # reward = modify_reward(next_state)
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state

            epoch_rewards = epoch_rewards + reward
            if agent.memory.memory_counter >= MEMORY_CAPACITY:
                agent.learn()
                if done: print(f'Epoch: {epoch:0=3d} | epoch_rewards:  {round(epoch_rewards, 2)}')
                # print(agent.memory.tree.tree[-MEMORY_CAPACITY:])
            if done:
                break
            state = next_state
