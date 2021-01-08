import torch
from torch.utils.tensorboard import SummaryWriter

x = torch.empty(5, 3)  # 空 tensor ,返回值为 内存中的 原来的值
# print(x)

x = torch.rand(5, 3)  # 均匀分布
print(x)

x = torch.zeros(5, 3, dtype=torch.long)  # 0值
print(x)

x = torch.tensor([5.5, 3])  # 指定值

writer = SummaryWriter('runs/fashion_mnist_experiment_1')
