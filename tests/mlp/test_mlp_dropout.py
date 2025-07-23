import unittest

import torch
from torch import nn
from d2l import torch as d2l

from tests.train import train_ch3


def dropout_layer(X, droppout):
    assert 0 <= droppout <= 1
    if droppout == 1:
        return torch.zeros_like(X)
    if droppout == 0:
        return X
    mask = (torch.rand(X.shape) > droppout).float()
    return mask * X / (1 - droppout)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
droupout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, droupout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

# 简洁实现
net2 = nn.Sequential(nn.Flatten(),
                     nn.Linear(784, 256),
                     nn.ReLU(),
                     # 在第一个全连接层之后添加一个dropout层
                     nn.Dropout(droupout1),
                     nn.Linear(256, 256),
                     nn.ReLU(),
                     # 在第二个全连接层之后添加一个dropout层
                     nn.Dropout(dropout2),
                     nn.Linear(256, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net2.apply(init_weights)


class TestMLPDropout(unittest.TestCase):
    def test_dropout_layer(self):
        X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
        print(X)
        print(dropout_layer(X, 0.))
        print(dropout_layer(X, 0.5))
        print(dropout_layer(X, 1.))

    def test_train(self):
        num_epochs, lr, batch_size = 10, 0.5, 256
        loss = nn.CrossEntropyLoss(reduction='none')
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
        trainer = torch.optim.SGD(net.parameters(), lr=lr)
        train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

    def test_train_concise(self):
        num_epochs, lr, batch_size = 10, 0.5, 256
        loss = nn.CrossEntropyLoss(reduction='none')
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
        trainer = torch.optim.SGD(net2.parameters(), lr=lr)
        train_ch3(net2, train_iter, test_iter, loss, num_epochs, trainer)
