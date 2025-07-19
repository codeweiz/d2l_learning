import unittest
import math
import time
import numpy as np
import torch
from d2l import torch as d2l


class TestVectorAccelerate(unittest.TestCase):
    def test_vector_accelerate(self):
        """测试矢量化加速"""
        n = 10000
        a = torch.ones([n])
        b = torch.ones([n])

        # 使用 Python 的 for 循环遍历向量：0.03660 sec
        c = torch.zeros(n)
        timer = Timer()
        for i in range(n):
            c[i] = a[i] + b[i]
        loop_time = timer.stop()
        print(f"for loop time: {loop_time:.5f} sec")

        # 使用矢量化操作：0.00001 sec
        timer.start()
        d = a + b
        vectorized_time = timer.stop()
        print(f"vectorized time: {vectorized_time:.5f} sec")

        # 验证结果循环比矢量化操作慢
        self.assertTrue(loop_time > vectorized_time)

    def test_normal_distribution(self):
        """测试正态分布"""
        # 再次使用numpy进行可视化
        x = np.arange(-7, 7, 0.01)
        # 均值和标准差对
        params = [(0, 1), (0, 2), (3, 1)]
        # 画图
        d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
                 ylabel='p(x)', figsize=(4.5, 2.5),
                 legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
        # 显示图片
        d2l.plt.show()

        # 可以发现 mu 值影响高度、sigma 影响水平位移


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


# 计算正态分布的概率密度函数
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


if __name__ == "__main__":
    unittest.main()
