"""
忽然感觉把memery拆出来真的香
"""
import torch
import torch.nn as nn
import numpy as np
import gym
import random

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


class Nstep_Memory(object):
    # memory buffer to store episodic memory
    def __init__(self, memory_size=MEMORY_CAPACITY, n_multi_step=1):
        self.memory = np.zeros(memory_size, dtype=object)

        self.memory_size = memory_size
        self.memory_counter = 0
        self.n_multi_step = n_multi_step

    def store_transition(self, transition):  # data = (state, action, reward, next_state, done) 4 1 1 4 1
        index = self.memory_counter % self.memory_size
        self.memory[index] = transition
        self.memory_counter = self.memory_counter + 1

    def sample(self, batch_size=BATCH_SIZE):
        if self.n_multi_step == 1:
            sample_index = np.random.choice(self.memory_size, batch_size)  # 2000中取一个batch index [batch_size]
            sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()  # [(s a r s) ] -> [(s) (a) (r) (s)]
            sample_state = torch.FloatTensor(np.vstack(sample_memory[0]))
            sample_action = torch.LongTensor(list(sample_memory[1])).view(-1, 1)
            sample_reward = torch.FloatTensor(list(sample_memory[2])).view(-1, 1)
            sample_next_state = torch.FloatTensor(np.vstack(sample_memory[3]))
            sample_done = torch.FloatTensor(sample_memory[4].astype(int)).view(-1, 1)
        else:
            # 2000中取一个batch index [batch_size--self.n_multi_step] 防止数据不够、
            sample_index = np.random.choice(self.memory_size-self.n_multi_step, batch_size)
            sample_index = (self.memory_counter+sample_index)%self.memory_size
            sample_memory = np.array(list(self.memory[sample_index]), dtype=object).transpose()  # [(s a r s) ] -> [(s) (a) (r) (s)]
            #如果刚好sample到最新的数据？要如何处理
            states, actions, rewards, next_states, dones = [], [], [], [], []
            for i in range(batch_size):
                finish = random.randint(self.n_multi_step, self.memory_size - 1)  # 2-2000 5
                begin = finish - self.n_multi_step  # 5 - 2 = 3
                sum_reward = 0  # n_step rewards
                data = self.buffer[begin:finish]
                state = data[0][0]
                action = data[0][1]
                for j in range(self.n_multi_step):
                    # compute the n-th reward
                    sum_reward += (self.gamma ** j) * data[j][2]
                    if data[j][4]:
                        states_look_ahead = data[j][3]
                        done_look_ahead = True
                        break
                    else:
                        states_look_ahead = data[j][3]
                        done_look_ahead = False

                states.append(state)
                actions.append(action)
                rewards.append(sum_reward)
                next_states.append(states_look_ahead)
                dones.append(done_look_ahead)

            return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones
        return sample_state, sample_action, sample_action, sample_reward, sample_next_state, sample_done

    def size(self):
        return len(self.buffer)


def weights_init(m):
    classname = m.__class__.__name__
    if classname == "Linear": m.weight.data.normal_(0.0, 0.1)


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.v_layers = nn.Sequential(
            nn.Linear(N_STATES, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )
        self.q_layers = nn.Sequential(
            nn.Linear(N_STATES, 10),
            nn.ReLU(),
            nn.Linear(10, N_ACTIONS),
        )
        self.apply(weights_init)  # initialization

    def forward(self, input):
        v = self.v_layers(input)  # 环境的价值
        q = self.q_layers(input)  # 动作的价值
        # actions_value = v.expand_as(q) + (q - q.mean(dim=1,keepdim=True).expand_as(q)) 等同与下方
        actions_value = v + (q - q.mean(dim=1, keepdim=True))
        return actions_value


class DuelingDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        if np.random.uniform() < EPSILON:  # greedy
            action = self.predict_action(state)
        else:  # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def predict_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # [4]=>[1 4]
        with torch.no_grad():
            actions_value = self.eval_net(state)  # [1 2]
            action = actions_value.max(1)[1].item()
            return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))  # 将每个参数打包起来 4+1+1+4  [10]
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter = self.memory_counter + 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter = self.learn_step_counter + 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 2000中取一个batch index [batch_size]
        sample_memory = self.memory[sample_index, :]
        sample_state = torch.FloatTensor(sample_memory[:, :N_STATES])  # [64 4]
        sample_action = torch.LongTensor(sample_memory[:, N_STATES:N_STATES + 1].astype(int))  # [batch 1]
        sample_reward = torch.FloatTensor(sample_memory[:, N_STATES + 1:N_STATES + 2])  # [batch 1]
        sample_next_state = torch.FloatTensor(sample_memory[:, -N_STATES:])  # [batch 4]

        # q_eval w.r.t the action in experience
        # input [batch_size dim_state] => output (batch, 2) [32,2]=>[32 1]=[batch_size action]
        q_eval = self.eval_net(sample_state).gather(1, sample_action)  # 去对应的acion的实际output
        # detach的作用就是不反向传播去更新，因为target的更新在前面定义好了的 [batch_size action]
        q_next = self.target_net(sample_next_state).detach()
        # Q = r + GAMMA*MAX(Q)
        q_target = sample_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1) TD MC?
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    m = Nstep_Memory(memory_size=10,n_multi_step=2)
    for i in range(10):
        m.store_transition((np.array([i, 2, 3, 4]), 10 + i, 100 + i, np.array([10 + i, 8, 9, 10]), True))
    print(m.memory)
    # print("--------------")
    print(m.sample(7))
