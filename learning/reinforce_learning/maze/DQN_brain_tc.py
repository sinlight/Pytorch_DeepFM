import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

# 定义命名元组 Transition 为环境中的单个转换，将 状态(state,action）: 结果(next_state,reward）
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# 循环缓冲区类别，保存最近观察到的转换，有点儿像dataset
class ReplayMemory(object):
    """ 缓存最近观察到的转换, 数据格式为 list 可以考虑使用 ndarray """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """将 Transition 加入缓存区"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """缓存区中 抽取 Transition """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
    """ 充当 Q-table 的作用"""

    def __init__(self, n_features, n_actions):
        """ 构造一个 2 层神经网络  """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_features, 10)
        self.fc2 = nn.Linear(10, n_actions)

    # 单元素或批次计算下次action
    def forward(self, x):  # todo 输入形状
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # todo 输出形状


class DQN(object):

    def __init__(self, n_actions, n_features, memory_size=500, batch_size=32,
                 learning_rate=0.01, gamma=0.9, e_greedy=0.9, e_greedy_increment=None,
                 target_update_step=10, output_graph=False, ):

        self.n_actions = n_actions
        self.n_features = n_features

        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        self.target_update_step = target_update_step  # 更换 target_net 的步数
        self.learn_step_counter = 0  # 记录学习次数 (用于判断是否更换 target_net 参数)

        # 创建缓存区
        self.memory = ReplayMemory(self.memory_size)

        # 创建 [policy_net, target_net]
        self.policy_net = MLP(self.n_actions, self.n_actions)
        self.target_net = MLP(self.n_actions, self.n_actions)

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            SummaryWriter("logs/", self.sess.graph)

        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看

    def store_transition(self, s, a, s_, r):
        """ 记录 一条 Transition

        Args:
            s: tensor 当前状态
            a: tensor 当前行为
            s_: tensor 下一步状态
            r: tensor 真实reward

        """
        self.memory.push(s, a, s_, r)

    def choose_action(self, state):
        """ 决策 一条 state 的行为

        Args:
            state: tensor 当前状态

        """
        if random.random() >= self.epsilon:  # 大于贪婪限制，随机筛选行为
            actions = torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long)  # size=(1,1)
        else:  # 小于贪婪限制，计算行为
            with torch.no_grad():
                actions = self.policy_net(state).max(1)[1].view(1, 1)  # tensor:shape(len(state))
        return actions

    def learn(self):

        if len(self.memory) < self.batch_size:
            return

        transition_list = self.memory.sample(self.batch_size)
        # a = zip(*transitions) 横向整体变成纵向；
        # Transition(*a)  # 将 a 拆解后 作为参数进入 Transition
        # 相当于从 dict 变成了 dataFrame
        batch_trans = Transition(*zip(*transition_list))  # 等价 Transition(*tuple(zip(*transitions)))

        # 判断 next_state 是否为空 布尔索引
        not_final_flag = torch.tensor(tuple(map(lambda s: s is not None, batch_trans.next_state)), dtype=torch.bool)

        # 将非空 next_state 的size(1,-1) list 拼接成 大size(n,-1)
        not_final_next_state_batch = torch.cat([s for s in batch_trans.next_state if s is not None], dim=0)

        state_batch = torch.cat(batch_trans.state, dim=0)  # size(n,n_features)
        action_batch = torch.cat(batch_trans.action, dim=0)  # size(n,1)
        reward_batch = torch.cat(batch_trans.reward, dim=0)  # size(n,1)

        # 计算 policy_net 得分 Q估计
        policy_net_out = self.policy_net(state_batch)  # 计算评估网络前馈 size(n,n_actions)
        # 以 action_batch 为索引，选择对应的得分
        state_action_values = policy_net_out.gather(dim=1, index=action_batch)  # tensor(n,1)

        # 计算下个状态的 target_net 得分
        next_state_values = torch.zeros(self.batch_size)  # 下个状态的最高得分
        target_net_out = self.target_net(not_final_next_state_batch)  # 计算事实网络前馈
        next_state_values[not_final_flag] = target_net_out.max(1)[0].detach()  # 获取并复制该状态的最优质，并取消追踪

        # 计算下一个状态的总得分 现实
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # tensor(n,1)

        # 计算损失函数 huber 损失
        criterion = F.smooth_l1_loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        # 定义损失函数和优化器
        optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 对参数值进行截取，相当于正则化
        optimizer.step()

        self.cost_his.append(loss)

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


def run_maze():
    step = 0
    for episode in range(300):
        # initial state
        state = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on state
            action = RL.choose_action(state)

            # RL take action and get next state and reward
            state_, reward, done = env.step(action)

            RL.store_transition(state, action, reward, state_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap state
            state = state_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    # RL.plot_cost()
