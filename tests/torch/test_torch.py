# 导入必要的库
import unittest  # Python 单元测试框架

import torch  # PyTorch 深度学习框架
import torch.nn as nn  # PyTorch 神经网络模块
import torch.nn.functional as F  # PyTorch 函数式接口（虽然在此代码中未使用）


# 自定义顺序容器模块，类似于 nn.Sequential
class CustomSequential(nn.Module):
    """
    自定义的顺序容器，用于按顺序执行多个神经网络层
    功能类似于 PyTorch 内置的 nn.Sequential
    """
    def __init__(self, *args):
        """
        初始化方法
        Args:
            *args: 可变数量的神经网络模块参数
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        # 遍历传入的所有模块，并将它们添加到当前模块中
        for idx, module in enumerate(args):
            # 使用索引作为模块名称，将每个模块注册到当前容器中
            # add_module 方法会将子模块正确注册，使其参数能被优化器发现
            self.add_module(str(idx), module)

    def forward(self, X):
        """
        前向传播方法
        Args:
            X: 输入张量
        Returns:
            经过所有子模块处理后的输出张量
        """
        # 遍历所有子模块，按顺序执行前向传播
        for module in self.children():
            X = module(X)  # 将上一层的输出作为下一层的输入
        return X


class TestTorch(unittest.TestCase):
    """
    PyTorch 测试类，继承自 unittest.TestCase
    用于测试自定义的 CustomSequential 模块
    """
    def test_torch(self):
        """
        测试方法：验证 CustomSequential 模块的功能
        """
        # 创建一个随机输入张量，形状为 (2, 20)
        # 2 表示批次大小（batch size），20 表示输入特征维度
        X = torch.rand(2, 20)
        print("X:", X)

        # 创建一个自定义的神经网络
        net = CustomSequential(
            nn.Linear(20, 256),  # 第一层：线性层，输入维度20，输出维度256
            nn.ReLU(),           # 第二层：ReLU激活函数，增加非线性
            nn.Linear(256, 10)   # 第三层：线性层，输入维度256，输出维度10
        )

        # 执行前向传播并打印结果
        # 输出形状应该是 (2, 10)：2个样本，每个样本10个输出特征
        print("\nnet(X):", net(X))
