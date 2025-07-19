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
