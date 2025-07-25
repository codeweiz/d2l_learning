import unittest

import torch
from torch import nn
from torch.nn import functional as F


# 自定义块
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


# 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


# 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight.data) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()



# 混合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


class TestModelConstruction(unittest.TestCase):
    def test_model_construction(self):
        net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        X = torch.rand(2, 20)
        print(net(X))

    def test_custom_model(self):
        net = MLP()
        X = torch.rand(2, 20)
        print(net(X))

    def test_sequential_block(self):
        net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
        X = torch.rand(2, 20)
        print(net(X))

    def test_fixed_hidden_mlp(self):
        net = FixedHiddenMLP()
        X = torch.rand(2, 20)
        print(net(X))

    def test_nest_mlp(self):
        chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
        X = torch.rand(2, 20)
        print(chimera(X))
