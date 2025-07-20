import unittest
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights)

# 损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


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


class TestSoftmaxRegressionConcise(unittest.TestCase):
    # 训练、预测
    def test_train(self):
        num_epochs = 10
        train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
        predict_ch3(net, test_iter)
