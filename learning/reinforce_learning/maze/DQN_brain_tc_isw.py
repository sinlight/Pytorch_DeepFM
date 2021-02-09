import random
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from maze_env import Maze

# 定义命名元组 Transition 为环境中的单个转换，将 状态(state,action）: 结果(next_state,reward）
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# with prioritized experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'is_weight'))

# 循环缓冲区类别，保存最近观察到的转换，有点儿像dataset
class ReplayMemory(object):
    """ 缓存最近观察到的转换 """

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
    """ 充当 Q-table 的作用 """

    def __init__(self, n_features, n_actions):
        """ 构造一个 3 层神经网络

        Args:
            n_features: 特征数量，输入的维度
            n_actions: 行为数量
        """

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_features, 8)
        self.fc2 = nn.Linear(8, 16)
        self.head = nn.Linear(16, n_actions)

    # 单元素或批次计算下次action
    def forward(self, input):
        """ 前向预测

        Args:
            input: size = (batch_size, n_feature)

        Returns:
            output: size = (batch_size, n_actions)
        """
        x = F.relu(self.fc1(input.float()))
        x = F.relu(self.fc2(x))
        output = self.head(x)
        return output


class Dueling_MLP(nn.Module):
    """ 充当 Q-table 的作用 """

    def __init__(self, n_features, n_actions):
        """ 构造 两个 3 层神经网络

        Args:
            n_features: 特征数量，输入的维度
            n_actions: 行为数量
        """

        super(Dueling_MLP, self).__init__()

        self.value_fc1 = nn.Linear(n_features, 8)
        self.value_fc2 = nn.Linear(8, 16)
        self.value_head = nn.Linear(16, 1)

        self.adv_fc1 = nn.Linear(n_features, 8)
        self.adv_fc2 = nn.Linear(8, 16)
        self.adv_head = nn.Linear(16, n_actions)

    # 单元素或批次计算下次action
    def forward(self, input):
        """ 前向预测

        Args:
            input: size = (batch_size, n_feature)

        Returns:
            output: size = (batch_size, n_actions)
        """
        # value # 专门分析 state 的 Value
        value = F.relu(self.value_fc1(input.float()))
        value = F.relu(self.value_fc2(value))
        value = self.value_head(value)  # n_features, 1

        # adv # 专门分析每种动作的 Advantage
        adv = F.relu(self.adv_fc1(input.float()))
        adv = F.relu(self.adv_fc2(adv))
        adv = self.adv_head(adv)  # n_features, n_actions

        # output Q = V(s) + A(s,a)
        # 合并 V 和 A, 为了不让 A 直接学成了 Q, 我们减掉了 A 的均值
        output = value + (adv - torch.mean(adv, axis=1, keepdim=True))

        return output


class DQN(object):

    def __init__(self, n_actions, n_features, memory_size=500, batch_size=32,
                 learning_rate=0.01, gamma=0.9, e_greedy=0.9, e_greedy_increment=None,
                 ddqn=False, dueling=False, target_update_step=10, output_graph=False, ):

        self.n_actions = n_actions
        self.n_features = n_features

        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.epsilon_increment = e_greedy_increment  # epsilon 的增量
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max  # 是否开启探索模式, 并逐步减少探索次数

        self.ddqn = ddqn
        self.dueling = dueling
        self.target_update_step = target_update_step  # 更换 target_net 的步数
        self.learn_step_counter = 0  # 记录学习次数 (用于判断是否更换 target_net 参数)

        # 创建缓存区
        self.memory = ReplayMemory(self.memory_size)

        # 创建 [policy_net, target_net]
        if self.dueling:
            self.policy_net = MLP(self.n_features, self.n_actions)
            self.target_net = MLP(self.n_features, self.n_actions)
        else:
            self.policy_net = Dueling_MLP(self.n_features, self.n_actions)
            self.target_net = Dueling_MLP(self.n_features, self.n_actions)

        # 输出 tensorboard 文件
        if output_graph:
            # $ tensorboard --logdir=logs
            SummaryWriter("logs/", self.sess.graph)

        self.cost_his = []  # 记录所有 cost 变化, 用于最后 plot 出来观看

    def store_transition(self, state, action, state_, reward, done):
        """ 记录 一条 Transition

        Args:
            state: 当前状态 需要 转换 tensor
            action: 当前行为 需要 转换 tensor
            state_: 下一步状态 需要 转换 tensor
            reward: 真实reward 需要 转换 tensor
            done: state_ 时候为结束

        """

        if type(state) is np.ndarray:
            state = torch.from_numpy(state).unsqueeze(0)  # (1, n_features)
        if type(state_) is np.ndarray:
            state_ = torch.from_numpy(state_).unsqueeze(0)  # (1, n_features)
        if type(action) is int:
            action = torch.tensor([action])  # (1)
        if type(reward) is int:
            reward = torch.tensor([reward])  # (1)
        if type(done) is bool:
            done = torch.tensor([done])  # (1)
        self.memory.push(state, action, state_, reward, done)

    def choose_action(self, state):
        """ 决策 一条 state 的行为

        Args:
            state: tensor 当前状态 size(1,n_feature)

        Returns:
            actions: size=(1)

        """
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).unsqueeze(0)  # (1, n_features)

        if random.random() >= self.epsilon:  # 大于贪婪限制，随机筛选行为
            actions = torch.tensor([random.randrange(self.n_actions)], dtype=torch.long)  # size=(1)
        else:  # 小于贪婪限制，计算行为
            with torch.no_grad():
                policy_net_out = self.policy_net(state)  # size = (1,n_actions)
                actions = policy_net_out.max(1).indices  # Size([1])
        return actions

    def learn(self):

        if len(self.memory) < self.batch_size:  # 数据量足够则学习，否则不学习
            return

        transition_list = self.memory.sample(self.batch_size)
        batch_trans = Transition(*zip(*transition_list))  # 等价 Transition(*tuple(zip(*transitions)))
        # a = zip(*transitions) 横向整体变成纵向；
        # Transition(*a)  # 将 a 拆解后 作为参数进入 Transition
        # 相当于从 dict 变成了 dataFrame

        # 判断 next_state 是否为空 布尔索引
        # not_final_flag = torch.tensor(tuple(map(lambda s: s is not None, batch_trans.next_state)), dtype=torch.bool)
        done_batch = torch.cat(batch_trans.done, dim=0)  # size(batch_size)
        not_final_flag = ~done_batch  # size(batch_size)

        # 将非空 next_state 的size(1,-1) list 拼接成 大size(n,-1)
        # not_final_next_state_batch = torch.cat([s for s in batch_trans.next_state if s is not None], dim=0)
        next_state_batch = torch.cat(batch_trans.state, dim=0)  # size(batch_size, n_features)
        not_final_next_state_batch = next_state_batch[not_final_flag]  # size(nf_batch_size, n_features)

        state_batch = torch.cat(batch_trans.state, dim=0)  # size(batch_size, n_features)
        action_batch = torch.cat(batch_trans.action, dim=0)  # size(batch_size)
        reward_batch = torch.cat(batch_trans.reward, dim=0)  # size(batch_size)

        # 计算 policy_net 得分 Q估计
        policy_net_out = self.policy_net(state_batch)  # 计算评估网络前馈 size(batch_size,n_actions)
        # 以 action_batch 为索引，选择对应的得分
        state_action_values = policy_net_out.gather(dim=1, index=action_batch.unsqueeze(0))  # 同index: (batch_size,1)

        # 计算下个状态的 target_net 得分
        next_state_values = torch.zeros(self.batch_size)  # size(batch_size)
        next_target_net_out = self.target_net(not_final_next_state_batch)  # size(nf_batch_size,n_actions)

        if not self.ddqn:  # Natural DQN
            next_state_values[not_final_flag] = next_target_net_out.max(1)[0].detach()  # size(batch_size)，赋值并取消追踪
        else:  # Double DQN
            with torch.no_grad():
                next_policy_net_out = self.policy_net(not_final_next_state_batch)  # size(nf_batch_size,n_actions)
                next_policy_net_action = next_policy_net_out.max(1)[1]  # size(batch_size)，预估网络选择行动

                # 目标网络选择指标size(batch_size, 1)
                next_target_net_value = next_target_net_out.gather(dim=1, index=next_policy_net_action.unsqueeze(1))
                next_state_values[not_final_flag] = next_target_net_value.squeeze(1)  # size(batch_size)，赋值

        # 计算下一个状态的总得分 现实
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  # size(batch_size)

        # 计算损失函数 huber 损失
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(0))

        # 定义损失函数和优化器
        optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # 对参数值进行截取，相当于正则化
        optimizer.step()

        # 逐渐增加 epsilon, 降低行为的随机性
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        self.cost_his.append(loss)
        self.learn_step_counter += 1  # 记录学习次数

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_step == 0:
            # 满足学习步数限制，更新target_net的参数
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # 输出学习过程中的 cost 变化曲线
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16, 4))
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def maze_update(env, RL, episodes=100, learn=True):
    # 学习 100 回合
    temp_step = 0
    for episode in range(episodes):
        # 初始化 state 的观测值
        state = env.reset()
        episode_step = 0
        while True:
            env.render()  # 更新可视化环境
            action = RL.choose_action(state)  # RL 大脑根据 state 的观测值挑选 action
            state_, reward, done = env.step(action)
            if learn:
                RL.store_transition(state, action, state_, reward, done)  # 记忆
                RL.learn()  # 学习 积累足够数据量了
            episode_step += 1
            state = state_
            if done:  # 回合结束，跳出本轮循环
                break
        if learn:
            print(episode, 'episode,', RL.learn_step_counter - temp_step, 'step_counter, 结束')
            temp_step = RL.learn_step_counter
        else:
            if reward == -1:
                print(episode, 'episode,', episode_step, 'step_counter, 失败结束')
            elif reward == 1:
                print(episode, 'episode,', episode_step, 'step_counter, 成功结束')
                # print(RL.policy_net.state_dict()['fc1.bias'])

    if not learn:
        print('测试结束')
    else:
        print('学习结束')
    env.destroy()

    return RL


if __name__ == '__main__':
    env = Maze('train')
    n_actions = env.n_actions
    n_features = env.n_features
    RL = DQN(n_actions, n_features, memory_size=256, batch_size=8,
             learning_rate=0.01, gamma=0.9, e_greedy=0.9, e_greedy_increment=None,
             ddqn=True, dueling=True, target_update_step=10, output_graph=False)

    print(RL.policy_net.state_dict()['fc1.bias'])
    RL = maze_update(env, RL, episodes=100, learn=True)
    # RL.plot_cost()

    print(RL.policy_net.state_dict()['fc1.bias'])
    env = Maze('test')
    RL = maze_update(env, RL, episodes=2, learn=False)

    from torch.utils.data import Dataset
