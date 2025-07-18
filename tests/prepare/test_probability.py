import unittest
import torch
from torch.distributions import multinomial
from d2l import torch as d2l


class TestProbability(unittest.TestCase):
    """概率"""

    def test(self):
        # 类似于投骰子，1 到 6 的概率都是 1/6
        fair_probs = torch.ones([6]) / 6

        # 多项分布，做 10 次，得出样本
        sample = multinomial.Multinomial(10, fair_probs).sample()
        print(sample)

        # 将结果存储为 32 位浮点数以进行除法，计算 1000 次投掷后，每个数字被投中了多少次
        counts = multinomial.Multinomial(1000, fair_probs).sample()
        print(counts / 1000)  # 相对频率作为估计值

        # 最终每个结果大约都是 0.167

        # 以 500 组实验，每组抽取 10 个样本
        counts = multinomial.Multinomial(10, fair_probs).sample((500,))
        print(counts.shape)  # torch.Size([500, 6])

        # 累计求和
        cum_counts = counts.cumsum(dim=0)
        print(cum_counts.shape)  # torch.Size([500, 6])

        # 计算相对频率
        estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
        print(estimates.shape)  # torch.Size([500, 6])

        # 绘制图像
        d2l.set_figsize((6, 4.5))
        for i in range(6):
            d2l.plt.plot(estimates[:, i].numpy(),
                         label=("P(die=" + str(i + 1) + ")"))
        d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
        d2l.plt.gca().set_xlabel('Groups of experiments')
        d2l.plt.gca().set_ylabel('Estimated probability')
        d2l.plt.legend()
        d2l.plt.show()
        # 每条实线对应于骰子的 6 个值中的一个，并给出骰子在每组实验后出现值的估计概率。
        # 当我们通过更多的实验获得更多的数据时，这 6 条实体曲线向真实概率收敛。
