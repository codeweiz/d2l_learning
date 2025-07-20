import unittest
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from d2l import torch as d2l

# 调用d2l库的use_svg_display函数，设置Matplotlib使用SVG格式显示图像
# 这在Jupyter Notebook或某些环境中可以提供更清晰的图像输出
d2l.use_svg_display()

# 定义全局batch_size为256，用于数据加载器的默认批量大小
# 这是一个合理的默认值，便于后续函数复用
batch_size = 256


# 定义函数get_dataloader_workers，返回数据加载器的worker进程数
# 使用6个进程来并行读取数据，以加速加载，尤其在多核CPU上
# 根据性能测试，6个进程是最优的（相比4、8、12个）
def get_dataloader_workers():
    return 6


# 定义函数get_fashion_mnist_labels，将数字标签转换为Fashion-MNIST的文本标签
# 输入：labels（数字标签列表）
# 输出：对应的文本标签列表
# Fashion-MNIST有10个类别，对应0-9的标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 定义函数show_images，用于绘制图像列表
# 参数：imgs（图像列表）、num_rows（行数）、num_cols（列数）、titles（可选标题列表）、scale（缩放比例，默认1.5）
# 支持PyTorch张量或PIL图像，隐藏轴刻度，并可选设置标题
# 返回axes数组，便于进一步操作
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 定义函数load_data_fashion_mnist，下载并加载Fashion-MNIST数据集
# 参数：batch_size（批量大小）、resize（可选，调整图像大小）
# 返回：训练和测试数据加载器的元组
# 使用transforms.Compose组合转换（可选Resize + ToTensor）
# 数据加载器支持shuffle（训练时）和多进程加载
# 优化：将worker数函数化，便于调整；添加download=True确保数据可用
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../../data", train=False, transform=trans, download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True,
                       num_workers=get_dataloader_workers()),
            DataLoader(mnist_test, batch_size, shuffle=False,
                       num_workers=get_dataloader_workers()))


# 定义单元测试类TestImageClassificationDataset，继承unittest.TestCase
# 用于测试数据集读取、加载和基本属性的正确性
class TestImageClassificationDataset(unittest.TestCase):

    # 测试方法test_read_dataset：验证数据集读取、长度、形状和加载性能
    # 创建数据集对象，断言样本数和形状
    # 获取小批量数据，可选显示图像（在测试中注释掉以避免阻塞）
    # 进行加载时间测试，使用Timer测量全数据集遍历时间
    # 优化：使用局部batch_size=18用于小批量测试；计时部分使用assert验证时间合理（例如<10秒）
    # 避免plt.show()在自动化测试中阻塞，改为non-blocking或注释
    def test_read_dataset(self):
        trans = transforms.ToTensor()
        mnist_train = torchvision.datasets.FashionMNIST(
            root="../../data", train=True, transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(
            root="../../data", train=False, transform=trans, download=True)
        self.assertEqual(len(mnist_train), 60000)
        self.assertEqual(len(mnist_test), 10000)

        self.assertEqual(mnist_train[0][0].shape, torch.Size([1, 28, 28]))
        self.assertEqual(mnist_test[0][0].shape, torch.Size([1, 28, 28]))

        train_loader = DataLoader(mnist_train, batch_size=18)
        X, y = next(iter(train_loader))

        # 显示图像
        show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
        d2l.plt.show(block=False)

        # 创建数据加载器用于性能测试，使用全局batch_size和6个workers
        # 使用d2l.Timer测量遍历整个数据集的时间
        # 根据先前测试，6个进程约5.90秒；这里添加assert确保时间<10秒（容忍变异）
        tran_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
        timer = d2l.Timer()
        for X, y in tran_iter:
            continue
        elapsed = timer.stop()
        print(f'{elapsed:.2f} sec')  # 打印时间，便于手动检查
        self.assertLess(elapsed, 10.0)  # 断言时间小于10秒，确保性能合理

    # 测试方法test_load_data：验证load_data_fashion_mnist函数
    # 加载调整大小后的数据集，检查批量形状和数据类型
    # 优化：添加断言验证预期形状（[32, 1, 64, 64] for X, [32] for y）和类型（float32, int64）
    # 避免仅print，改为使用assert进行自动化验证
    def test_load_data(self):
        train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
        for X, y in train_iter:
            # 打印
            print(X.shape, X.dtype, y.shape, y.dtype)

            # 断言验证形状和类型
            self.assertEqual(X.shape, torch.Size([32, 1, 64, 64]))
            self.assertEqual(X.dtype, torch.float32)
            self.assertEqual(y.shape, torch.Size([32]))
            self.assertEqual(y.dtype, torch.int64)
            break  # 只检查第一个批量


# 主程序入口：如果脚本作为主程序运行，执行所有单元测试
if __name__ == '__main__':
    unittest.main()
