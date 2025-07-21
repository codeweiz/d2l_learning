import unittest

import torch
from torch import nn
from d2l import torch as d2l

from tests.train import train_ch3, predict_ch3

# 两个全连接层，784 到 256，256 到10
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))


# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


class TestMLPConcise(unittest.TestCase):
    # 高级 API 实现的多层感知机训练与预测
    def test_mlp_train(self):
        train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
        predict_ch3(net, test_iter)
