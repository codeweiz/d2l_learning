import unittest

import torch
from torch import nn


# 初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里(1, 1)表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


class TestPaddingStrides(unittest.TestCase):
    def test_padding(self):
        # 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
        conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        X = torch.rand(size=(8, 8))
        # 8 + 2 - 3 + 1 = 8
        assert comp_conv2d(conv2d, X).shape == (8, 8)

        # 8 + 4 - 5 + 1 = 8
        # 8 + 2 - 3 + 1 = 8
        conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
        assert comp_conv2d(conv2d, X).shape == (8, 8)

    def test_stride(self):
        X = torch.rand(size=(8, 8))
        conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
        # (8 + 2 - 3 + 1) / 2 = 4
        print(comp_conv2d(conv2d, X).shape)
        assert comp_conv2d(conv2d, X).shape == (4, 4)

        X = torch.rand(size=(8, 8))
        conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
        # (8 + 2 * 0 - 3 + 1) / 3 = 2
        # (8 + 2 * 1 - 5 + 1) / 4 = 2
        assert comp_conv2d(conv2d, X).shape == (2, 2)


