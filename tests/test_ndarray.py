import torch


# 入门
def test_begin():
    # arange：使用 arange 创建一个行向量 x，这个行向量包含以0开始的前12个整数，它们默认创建为整数。
    x = torch.arange(12)
    print(x)

    # reshape：通过张量的 shape 属性来访问张量的形状，将张量 x 从形状为（12,）的行向量转换为形状为（3,4）的矩阵
    X = x.reshape(3, 4)
    print(X)

    # reshape -1:在知道宽度或高度后，使用 -1 就可以自动计算出维度
    print(x.reshape(-1, 4))
    print(x.reshape(3, -1))

    # zeros、ones：使用 全0、全1 来初始化矩阵，形状为 (2,3,4)，从外到内的元素个数为 2、3、4
    print(torch.zeros((2, 3, 4)))
    print(torch.ones((2, 3, 4)))

    # randn：从特定的概率分布中随机采样来得到张量中每个元素的值，均值为0、标准差为1的标准高斯分布（正态分布），形状为 (3,4)
    print(torch.randn(3, 4))

    # tensor：通过提供包含数值的 Python 列表（或嵌套列表），来为所需张量中的每个元素赋予确定值
    print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))


# 运算符
def test_calculator():
    # + - * / **：对于任意具有相同形状的张量，常见的标准算术运算符（`+`、`-`、`*`、`/`和`**`）都可以被升级为按元素运算
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y, x - y, x * y, x / y, x ** y)

    # exp：y = e ^ x
    print(torch.exp(x))

    # cat：张量连结 concatenate，dim=0 表示按行连结，dim=1 表示按列连结
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(torch.cat((X, Y), dim=0))
    print(torch.cat((X, Y), dim=1))

    # ==：逻辑运算符，X == Y 表示 X 和 Y 在每个位置相等，则新张量中相应项的值为1
    print(X == Y)

    # sum：对张量中的所有元素进行求和，会产生一个单元素张量
    print(X.sum())


# 广播机制
def test_broadcast():
    # 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状
    # 这里一个是（3，1），一个是（1，2），广播后就统一为了（3，2），用 0 补齐
    a = torch.arange(3).reshape((3, 1))
    b = torch.arange(2).reshape((1, 2))
    print(a, b)
    print(a + b)


# 索引和切片
def test_index_slice():
    # 索引：第一个元素的索引是 0，最后一个元素的索引是 -1，1:3 表示选择第2个和第3个元素
    # X[-1] 表示选择最后一行：[8, 9, 10, 11]
    # X[1:3] 表示选择第2行和第3行：[[4, 5, 6, 7], [8, 9, 10, 11]]
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    print(X)
    print(X[-1], X[1:3])

    # 切片：X[:, 2] 表示选择所有行的第3列，X[:, 2:3] 表示选择所有行的第3列，但保留二维结构
    # : 在前表示所有行，在后表示所有列
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    print(X[:, 2])
    print(X[:, 2:3])
    print(X[:, 2:3].shape)

    # 赋值：为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值
    # X[0:2, :] 表示选择第1行和第2行的所有列
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    X[0:2, :] = 12
    print(X)


# 节省内存
def test_save_memory():
    # 为新结果分配内存，使用 Y = Y + X，会导致为结果分配新的内存，可以发现 第一个 Y 和第二个 Y 的内存地址不同
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(id(Y))
    Y = Y + X
    print(id(Y))

    # 使用切片表示法，可以原地执行，使用 Y[:] = Y + X，不会为结果分配新的内存，可以发现 Y 的内存地址不变
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(id(Y))
    Y[:] = Y + X
    print(id(Y))

    # 使用 X += Y，不会为结果分配新的内存，可以发现 X 的内存地址不变
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(id(X))
    X += Y
    print(id(X))


# 转换为其他Python对象
def test_convert():
    # numpy：转换为 NumPy 张量、tensor：转换为 torch 张量
    X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    A = X.numpy()
    B = torch.tensor(A)
    print(type(A), type(B))
    print(A, B)

    # item：将大小为1的张量转换为Python标量
    a = torch.tensor([3.5])
    print(a, a.item(), float(a), int(a))
