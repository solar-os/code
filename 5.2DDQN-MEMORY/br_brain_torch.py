import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, n_actions)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class DQN:
    def __init__(self, n_states, n_actions):
        print("<DQN init>")
        # DQN有两个net:target_net 和eval_net,具有选择动作、存经历、学习的功能
        # 创建target_net和eval_net
        self.target_net, self.eval_net = Net(n_states, n_actions), Net(n_states, n_actions)
        self.loss = nn.MSELoss()  # 损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.n_actions = n_actions
        self.n_states = n_states

        self.learn_step_counter = 0  # target_net网络学习计数
        self.memory_counter = 0  # 记忆计数
        self.memory = np.zeros((100, self.n_states * 2 + 2))
        self.cost = []
        self.reward = []

    def choose_action(self, state, epsilon):  # 根据当前状态选择动作
        # 传过来的状态是一维的，我们要将其转换为2维的
        state = torch.unsqueeze(torch.FloatTensor(state), 0)
        if np.random.uniform() < epsilon:  # 90%的概率选择评估网络输出的最大值
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
            # print(action, actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
            # print(action)
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % 100
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target_net每两百次更新一次参数（不会及时更新参数），用于预测
        if self.learn_step_counter % 100 == 0:   # 改成每100次更新一次目标网络
            self.target_net.load_state_dict((self.eval_net.state_dict()))
        self.learn_step_counter += 1

        sample_index = np.random.choice(100, 16)
        memory = self.memory[sample_index, :]
        state = torch.FloatTensor(memory[:, :2])
        action = torch.LongTensor(memory[:, 2:3])
        reward = torch.LongTensor(memory[:, 3:4])
        next_state = torch.FloatTensor(memory[:, 4:6])

        # 计算损失，q_eval是对动作的预测，q_target是对动作的实际值
        q_eval = self.eval_net(state).gather(1, action)
        q_next = self.target_net(next_state).detach()
        q_target = reward + 0.9 * q_next.max(1)[0].unsqueeze(1)
        loss = self.loss(q_eval, q_target)
        self.cost.append(loss)
        # 反向传播
        self.optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        self.optimizer.step()

    def test_value(self, state):
        q_eval = self.eval_net.forward(state)
        # q_target = self.target_net.forward(state)
        # print(q_eval)
        action = torch.max(q_eval, 1)[1].data.numpy()[0]
        return action, q_eval

    def store_model(self):
        torch.save(self.eval_net.state_dict(), 'eval_net_DQN.pt')

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost)), self.cost)
        plt.xlabel("step")
        plt.ylabel("cost")
        plt.show()

    def plot_reward(self, reward):
        inter = 1000
        n = len(reward)
        reward = reward[40000: n: inter]
        plt.plot(reward)
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.show()
