import unittest
import numpy as np
from matplotlib_inline import backend_inline
from d2l import torch as d2l


def f(x):
    """函数：f(x) = 3x^2 - 4x"""
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    """通过数值方法逼近导数：h 趋于 0 时，(f(x+h) - f(x)) / h 的极限"""
    return (f(x + h) - f(x)) / h


def use_svg_display():
    """使用 SVG 格式显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """设置 Matplotlib 图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置 Matplotlib 的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    def has_one_axis(data):
        """检查数据是否只有一个轴"""
        return (hasattr(data, "ndim") and data.ndim == 1 or
                isinstance(data, list) and not hasattr(data[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    # 返回 axes 以便进一步操作（优化：原代码不返回，现在添加返回以便测试）
    return axes


class TestCalculus(unittest.TestCase):
    def test_derivative(self):
        """测试数值导数逼近真实导数（在 x=1 时应接近 2）"""
        expected_derivative = 2.0  # f'(1) = 6*1 - 4 = 2
        h = 0.1
        for i in range(5):
            num_lim = numerical_lim(f, 1, h)
            print(f'h={h:.5f}, numerical limit={num_lim:.5f}')
            # 断言：随着 h 减小，数值极限应越来越接近 2
            self.assertAlmostEqual(num_lim, expected_derivative, places=1 if i < 2 else 4)
            h *= 0.1

    def test_plot(self):
        """测试绘图函数"""
        x = np.arange(0, 3, 0.1)
        axes = plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
        # 简单断言：检查 axes 是否被正确设置
        self.assertIsNotNone(axes.get_xlabel())
        self.assertIsNotNone(axes.get_ylabel())
        # 显示图表
        d2l.plt.show()


if __name__ == '__main__':
    unittest.main()
