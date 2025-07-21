import unittest
import torch
from d2l import torch as d2l


class TestMultilayerPerceptrons(unittest.TestCase):
    # 修正线性单元：Rectified Linear Unit，提供了一种非常简单的非线性变换，给定元素x，ReLU函数被定义为该元素与0的最大值
    # ReLU(x) = max(x, 0)，限制最小值为 0
    def test_relu(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.relu(x)
        d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
        d2l.plt.show()

    def test_relu2(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.relu(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
        d2l.plt.show()

    # 挤压函数，将范围(-inf, inf)中的任意输入压缩到区间(0, 1)中的某个值
    def test_sigmoid(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.sigmoid(x)
        d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
        d2l.plt.show()

    def test_sigmoid2(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.sigmoid(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
        d2l.plt.show()

    # 双曲正切函数，将范围(-inf, inf)中的任意输入压缩到区间(-1, 1)中的某个值
    def test_tanh(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.tanh(x)
        d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
        d2l.plt.show()

    def test_tanh2(self):
        x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
        y = torch.tanh(x)
        y.backward(torch.ones_like(x), retain_graph=True)
        d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
        d2l.plt.show()
