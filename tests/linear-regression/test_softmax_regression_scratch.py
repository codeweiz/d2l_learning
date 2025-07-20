import unittest

# 导入PyTorch库，这是深度学习的核心框架，用于处理张量（类似于多维数组）和构建神经网络
# torch提供自动求导、优化等功能，非常适合初学者从零开始学习
import torch

# 导入d2l库（来自《动手学深度学习》书籍的配套工具包）
# d2l包含了许多实用函数，如数据加载、可视化工具，帮助简化深度学习实验
from d2l import torch as d2l

# 定义全局变量batch_size为256
# batch_size是“批量大小”的意思：在训练模型时，我们不是一次处理所有数据，而是分成小批量（这里是256个样本）
# 为什么用批量？因为全数据训练太慢，批量可以加速计算并使用GPU更高效；256是一个常见选择，平衡了速度和内存使用
batch_size = 256

# 使用d2l库的load_data_fashion_mnist函数加载Fashion-MNIST数据集
# Fashion-MNIST是一个图像分类数据集，包含10类服装图片（如T恤、鞋子），每个图片是28x28灰度图像
# 这个函数会自动下载数据、应用转换（例如将图片转为张量），并返回训练数据迭代器（train_iter）和测试数据迭代器（test_iter）
# 迭代器就像一个“数据流”，可以方便地一批一批地取出数据用于训练或测试
# 为什么分训练和测试？训练数据用于学习模型，测试数据用于评估模型在未见过数据上的表现，避免过拟合（模型只记住训练数据而不泛化）
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
# num_inputs = 784：输入维度，因为每个图像是28x28像素，展平后变成784个数字（就像把图片拉成一条线）
# num_outputs = 10：输出维度，因为有10个类别（Fashion-MNIST的服装类型）
# 模型本质上是线性回归，但输出经过softmax转为概率
num_inputs = 784
num_outputs = 10

# W是权重矩阵：形状为(784, 10)，每个输入特征连接到10个输出
# 使用torch.normal初始化为小随机值（均值0，标准差0.01），这是常见初始化方法，避免模型从零开始太慢
# requires_grad=True：告诉PyTorch这个变量需要计算梯度（用于后续优化）
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)

# b是偏置向量：形状为(10,)，初始化为零
# 偏置帮助模型捕捉数据中的偏移
b = torch.zeros(num_outputs, requires_grad=True)


# 定义 softmax 函数：softmax是一种激活函数，用于多分类问题，将模型的原始输出（logits）转换为概率分布
# 为什么需要softmax？模型输出的是原始分数（如3.2, 1.5, -0.1），softmax把它们转为正概率（如0.7, 0.2, 0.1），且所有概率加起来=1
# 这有助于解释模型的“信心”：哪个类别概率最高就是预测结果
# 函数步骤：
# 1. 对输入X的每个元素取指数（exp），确保所有值正
# 2. 对每行（每个样本）求和，得到规范化常数（partition）
# 3. 每个元素除以其行的常数，确保每行概率和为1
# 这里X通常是(batch_size, num_outputs)的矩阵，广播机制允许高效计算
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


# 定义模型 net：这是softmax回归模型的核心
# 输入X是图像批量，形状如(batch_size, 1, 28, 28)
# 先用reshape展平为(batch_size, 784)的向量（-1表示自动推断批量大小）
# 然后计算线性变换：X * W + b，得到原始输出（logits）
# 最后应用softmax转为概率
# 这个模型很简单，没有隐藏层，适合初学者理解线性分类
def net(X):
    # 数据传递到模型之前，使用 reshape 函数将每张原始图像展平为向量
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 定义损失函数：交叉熵损失（cross_entropy）
# 损失函数衡量模型预测(y_hat)和真实标签(y)的差距
# 交叉熵计算：对于每个样本，取其真实类别的预测概率，取负对数（-log(p)）
# 为什么负对数？如果概率p接近1（正确），损失小；如果p小，损失大，鼓励模型提高正确概率
# 这里y_hat是概率矩阵，y是标签索引，使用range(len(y_hat))选择每个样本的行
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


# 定义准确率函数：计算预测正确的数量
# y_hat可能是概率矩阵，先用argmax取每行最大概率的索引作为预测标签
# 然后比较与真实y是否相等，求和返回正确数（float类型）
# 这是一个评估指标，不是损失，用于人类理解模型好坏
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# Accumulator类：一个简单工具类，用于累加多个变量（如总损失、总正确数、总样本数）
# 在训练中，我们需要汇总整个数据集的指标，这个类帮助逐步累加
# __init__：初始化n个0.0
# add：添加值到对应位置
# reset：重置为零
# __getitem__：像列表一样访问
# 这避免了手动用变量累加，代码更干净
class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# evaluate_accuracy函数：计算模型在给定数据集（如测试集）上的整体准确率
# 如果net是PyTorch的nn.Module，设置为eval模式（关闭dropout等训练行为）
# 使用Accumulator累加总正确数和总样本数
# with torch.no_grad()：评估时不需要计算梯度，节省内存和时间
# 遍历数据迭代器，计算每个批量的准确率，累加，最后返回平均准确率（正确数 / 总样本）
# 这用于监控模型性能，而不参与训练
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# train_epoch_ch3函数：训练模型一个完整迭代周期（epoch，即遍历一次全部训练数据）
# 如果net是nn.Module，设置为train模式（启用训练行为）
# 使用Accumulator累加总损失、总正确数、总样本
# 对于每个批量：
#   - 前向传播：y_hat = net(X)
#   - 计算损失 l
#   - 如果updater是PyTorch优化器：清零梯度、求平均损失的反向、步进更新
#   - 否则：求总损失的反向、调用自定义updater
# 累加指标，最后返回平均损失和平均准确率
# epoch是训练的基本单位，通常跑多个epoch让模型逐步改善
def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


# 定义学习率lr为0.1
# 学习率控制每次参数更新的步长：太大可能不收敛，太小训练慢
# 0.1是softmax回归的常见起点，可通过实验调整
lr = 0.1


# updater函数：定义如何更新参数，使用d2l.sgd（随机梯度下降）
# sgd基于梯度（损失对参数的导数）调整W和b：参数 -= lr * 梯度 / batch_size
# 这是一个自定义优化器，适合从零实现理解梯度下降
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


# train_ch3函数：完整训练过程，运行多个epoch
# 创建Animator来实时绘制训练曲线（损失、准确率）
# 每个epoch：调用train_epoch_ch3训练，评估测试准确率，添加数据到动画
# 最后使用assert检查训练是否成功（损失<0.5，准确率>0.7），如果不满足会报错
# 这整合了训练循环，帮助监控过拟合（训练acc高但测试acc低）
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# predict_ch3函数：使用训练好的模型预测测试数据的标签并可视化
# 从test_iter取一个批量，计算预测（argmax取最高概率类别）
# 使用d2l函数获取文本标签，创建标题（真实 + 预测）
# 显示前n个图像及其标签，帮助直观检查模型错误
# 这是一个推理（inference）步骤，用于看到模型的实际输出
def predict_ch3(net, test_iter, n=6):
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()


# Animator类：用于在训练过程中动态绘制图表（如损失曲线）
# 初始化时设置轴标签、范围、图例等
# add方法：添加新数据点（x如epoch，y如损失），清除旧图，重绘所有线
# 这帮助可视化训练进展：损失是否下降？准确率是否上升？
# 在非Jupyter环境中，使用fig.show()显示
class Animator:
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.show()
        # display.display(self.fig)
        # display.clear_output(wait=True)


# TestSoftmaxRegressionScratch类：使用unittest框架测试代码的各个部分
# 每个方法是一个测试用例，运行时会自动执行
# 这帮助验证函数是否正确工作，尤其在零基础学习时，可以逐步调试
class TestSoftmaxRegressionScratch(unittest.TestCase):
    # 测试 softmax 操作
    # 创建随机输入，计算softmax，打印结果和每行和（应接近[1,1]）
    # 这验证softmax是否正确转换为概率
    def test_softmax(self):
        X = torch.normal(0, 1, (2, 5))
        X_prob = softmax(X)
        print(X_prob, X_prob.sum(1))

    # 测试损失函数
    # 使用示例y_hat（概率）和y（标签），打印选择的概率和损失
    # 也计算并打印平均准确率（这里应为0.5）
    # 这帮助理解损失如何量化错误
    def test_cross_entropy(self):
        y = torch.tensor([0, 2])
        y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
        print(y_hat[[0, 1], y])
        print(cross_entropy(y_hat, y))

        # 测试分类精度：0.5
        print(accuracy(y_hat, y) / len(y))

    # 测试指定数据集上模型的精度
    # 计算并打印未训练模型在测试集上的准确率（随机猜测约0.1）
    # 这显示初始模型的表现
    def test_evaluate_accuracy(self):
        print(evaluate_accuracy(net, test_iter))

    # 测试训练
    # 设置10个epoch，调用train_ch3运行完整训练过程
    # 会显示动画曲线和最终assert
    def test_train(self):
        num_epochs = 10
        train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    # 测试直接预测，没有训练
    # 使用初始模型预测，显示结果（许多错误，准确率低）
    # 这对比训练前后差异
    def test_predict(self):
        predict_ch3(net, test_iter)

    # 测试训练后预测
    # 先训练10个epoch，然后预测，显示结果（准确率应高）
    # 这演示完整流程：训练 + 推理
    def test_train_and_predict(self):
        num_epochs = 10
        train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
        predict_ch3(net, test_iter)
