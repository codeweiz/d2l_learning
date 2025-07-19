import random
import unittest

import torch
from d2l import torch as d2l


# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 根据批量大小、特征矩阵、标签向量生成小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 随机读取样本，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 获取随机的权重和偏置
def get_params(num_inputs):
    w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


class TestLinearRegression(unittest.TestCase):
    # 测试生成数据集
    def test_synthetic_data(self):
        w = torch.tensor([2, -3.4])
        b = 4.2
        num_examples = 1000
        X, y = synthetic_data(w, b, num_examples)
        print('features:', X[0], '\nlabel:', y[0])
        self.assertEqual(X.shape, (num_examples, 2))
        self.assertEqual(y.shape, (num_examples, 1))

        # 生成第二个特征 features[:, 1] 和 labels 的散点图
        d2l.set_figsize()
        d2l.plt.scatter(X[:, 1].detach().numpy(), y.detach().numpy(), 1)
        d2l.plt.show()

    # 测试生成小批量数据
    def test_data_iter(self):
        batch_size = 10
        features, labels = synthetic_data(torch.tensor([2, -3.4]), 4.2, 1000)
        for X, y in data_iter(batch_size, features, labels):
            print(X, '\n', y)
            break
        # 当运行迭代时，会连续地获取不同的小批量，直至遍历完整个数据集

    # 训练
    def test_train(self):
        lr = 0.03
        num_epochs = 3
        net = linreg
        loss = squared_loss
        batch_size = 10

        # 生成数据集
        true_w = torch.tensor([2, -3.4])
        true_b = 4.2
        features, labels = synthetic_data(true_w, true_b, 1000)

        # 初始化模型参数
        w, b = get_params(2)

        for epoch in range(num_epochs):
            for X, y in data_iter(batch_size, features, labels):
                l = loss(net(X, w, b), y)  # X 和 y 的小批量损失
                # 因为 l 的形状是 (batch_size, 1)，而不是一个标量。l 中的 所有元素被加到一起， 并以此计算关于 [w, b] 的梯度
                l.sum().backward()
                sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

        # 比较真实参数和估计参数
        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - b}')
