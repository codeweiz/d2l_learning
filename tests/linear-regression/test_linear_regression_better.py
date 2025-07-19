import unittest

import torch
from d2l import torch as d2l
from torch import nn
from torch.utils import data

# 真实的权重与偏置，以下线性回归就是为了训练出与真实权重和偏置误差尽可能小的权重和偏置
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# 生成数据集，包含特征和标签
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    """构造一个 PyTorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10

# 定义模型：输入特征性状为2，输出特征形状为1
net = nn.Sequential(nn.Linear(2, 1))

# 定义损失函数：平方 L2 范数
loss = nn.MSELoss()

# 定义优化算法：小批量随机梯度下降
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


class TestLinearRegressionBetter(unittest.TestCase):
    # 训练
    def test_train(self):
        num_epochs = 3
        for epoch in range(num_epochs):
            for X, y in load_array((features, labels), batch_size):
                l = loss(net(X), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            l = loss(net(features), labels)
            print(f'epoch {epoch + 1}, loss {l:f}')

        # 比较真实参数和估计参数
        print(f'w的估计误差: {true_w - net[0].weight.data}')
        print(f'b的估计误差: {true_b - net[0].bias.data}')
