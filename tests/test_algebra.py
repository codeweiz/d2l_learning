import unittest
import torch


class TestAlgebra(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    # 标量：由只有一个元素的张量表示
    def test_scalar(self):
        x = torch.tensor(3.0)
        y = torch.tensor(2.0)

        print(x + y, x * y, x / y, x ** y)

    # 向量
    def test_vector(self):
        x = torch.arange(4)
        print(x)

        # 访问向量的任一元素
        print(x[3])

        # 向量只是一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此
        print(len(x))
        print(x.shape)

    # 矩阵
    def test_matrix(self):
        A = torch.arange(20).reshape(5, 4)
        print(A)

        # 访问矩阵的任一元素
        print(A[2, 3])

        # 矩阵的转置
        print(A.T)

        # 矩阵的加法
        B = A.clone()
        print(A + B)

        # 矩阵的乘法
        print(A * B)

    # 张量
    def test_tensor(self):
        X = torch.arange(24).reshape(2, 3, 4)
        print(X)

        # 访问张量的任一元素
        print(X[1, 2, 3])

        # 张量只是一个数字数组，就像每个数组都有一个长度一样，每个张量也是如此
        print(len(X))
        print(X.shape)

    # 张量算法的基本性质
    def test_tensor_algorithm(self):
        A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        B = A.clone()

        # 两个相同形状的张量，任何按元素二元运算的结果都将是相同形状的张量
        print(A + B)
        print(A * B)

        # 任何按元素的一元运算都不会改变其操作数的形状
        print(A.shape, A.sqrt().shape)

        # 张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘
        print((A * 2).shape, (A + 2).shape)

    # 降维
    def test_reduce(self):
        X = torch.arange(4, dtype=torch.float32)
        print(X, X.sum())

        # axis=0 表示按轴0降维，即按列降维
        # axis=1 表示按轴1降维，即按行降维
        # axis=[0, 1] 表示按轴0和轴1降维，即按列和行降维，等价于对所有元素求和
        A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        print(A, A.sum(), A.sum(axis=0), A.sum(axis=1), A.sum(axis=[0, 1]))

        # 平均值
        print(A.mean(), A.sum() / A.numel())

        # 计算平均值也可以沿指定轴降低张量的维度
        print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])
        print(A.mean(axis=1), A.sum(axis=1) / A.shape[1])

        # 非降维求和，保持轴数不变
        print(A.sum(axis=1, keepdims=True))
        print(A / A.sum(axis=1, keepdims=True))

        # 累加求和
        print(A.cumsum(axis=0))

    # 点积：相同位置的按元素乘积的和
    def test_dot_product(self):
        x = torch.arange(4, dtype=torch.float32)
        y = torch.ones(4, dtype=torch.float32)
        print(torch.dot(x, y))

        # 等价于
        print(torch.sum(x * y))

    # 矩阵-向量积：(5,4) mv (1, 4) = (1, 4)
    def test_matrix_vector_product(self):
        A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        x = torch.ones(4, dtype=torch.float32)
        print(A, x, torch.mv(A, x))

        # 等价于
        print(A @ x)

    # 矩阵-矩阵乘法：（n, k）mm (k,m) = (n,m)
    def test_matrix_matrix_multiplication(self):
        A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        B = torch.ones(4, 3)
        print(A, B, torch.mm(A, B))

        # （5，4）mm （4，5） = （5，5）
        C = A.clone().T
        print(A, C, torch.mm(A, C))

    # 范数
    def test_norm(self):
        u = torch.tensor([3.0, -4.0])
        print(torch.norm(u))
        print(torch.abs(u).sum())
        print((u ** 2).sum() ** 0.5)

        # L1 范数：向量元素绝对值之和
        print(torch.abs(u).sum())

        # L2 范数：欧几里得距离就是 L2 范数，即向量元素平方和的平方根
        print(torch.norm(u))
        print(torch.sqrt((u ** 2).sum()))

        # Frobenius 范数：矩阵元素平方和的平方根
        print("Frobenius 范数：", torch.norm(torch.ones(4, 9)))


#
"""
小结：

标量、向量、矩阵和张量是线性代数中的基本数学对象。
向量泛化自标量，矩阵泛化自向量。
标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴。
一个张量可以通过 sum 和 mean 沿指定的轴降低维度。
两个矩阵的按元素乘法被称为他们的 Hadamard 积。它与矩阵乘法不同。
在深度学习中，我们经常使用范数，如 L1范数、L2范数 和 Frobenius范数。
我们可以对标量、向量、矩阵和张量执行各种操作。
"""
