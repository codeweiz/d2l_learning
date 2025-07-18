import time
import unittest

import torch


class TestAutograd(unittest.TestCase):
    def test_autograd(self):
        """测试基本自动微分"""
        x = torch.arange(4.0, requires_grad=True)

        # 计算 y = 2 * (x^T * x)
        y = 2 * torch.dot(x, x)
        y.backward()
        print(f"x.grad after first backward: {x.grad}")

        # 验证梯度应为 4 * x
        expected_grad = 4 * x
        print(f"Expected grad (4 * x): {expected_grad}")
        self.assertTrue(torch.allclose(x.grad, expected_grad))

        # 清除梯度
        x.grad.zero_()

        # 计算 y = x.sum()，其梯度应为 [1, 1, 1, 1]
        y = x.sum()
        y.backward()
        print(f"x.grad after sum backward: {x.grad}")

        # 验证梯度
        expected_grad = torch.ones_like(x)
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_autograd_non_scalar(self):
        """测试非标量变量的反向传播"""
        x = torch.arange(4.0, requires_grad=True)
        y = x * x  # y 是向量: [0, 1, 4, 9]

        # 等价于 y.backward(torch.ones_like(y))
        y.sum().backward()
        print(f"x.grad: {x.grad}")

        # 验证梯度应为 2 * x（因为 y_i = x_i^2 的导数是 2 * x_i）
        expected_grad = 2 * x
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_autograd_detach(self):
        """测试分离计算（detach）"""
        x = torch.arange(4.0, requires_grad=True)
        y = x * x  # y: [0, 1, 4, 9]

        # 分离 y，u 与 y 值相同，但不参与梯度计算
        u = y.detach()
        print(f"u (detached y): {u}")

        # 计算 z = u * x，其梯度只关于 x（u 被视为常量）
        z = u * x
        z.sum().backward()
        print(f"x.grad: {x.grad}")

        # 验证梯度应为 u（因为 z_i = u_i * x_i，u_i 是常量，导数是 u_i）
        expected_grad = u
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_autograd_control_flow(self):
        """测试 Python 控制流的梯度计算（autograd 支持条件分支）"""
        x = torch.arange(4.0, requires_grad=True)

        # 控制流：逐元素条件赋值（x < 2 时 y = 2*x，否则 y = 4*x）
        # 结果 y: [0*2=0, 1*2=2, 2*4=8, 3*4=12]
        y = torch.where(x < 2, 2 * x, 4 * x)
        print(f"y after where: {y}")

        # 计算 z = y.sum()，反向传播
        # 预期梯度：根据条件路径，<2 时导数=2，否则=4 → [2, 2, 4, 4]
        z = y.sum()
        z.backward()
        print(f"x.grad: {x.grad}")

        # 验证梯度
        expected_grad = torch.where(x < 2, torch.tensor(2.0), torch.tensor(4.0))
        print(f"Expected grad: {expected_grad}")
        self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_autograd_dynamic_control_flow(self):
        """测试动态控制流（循环 + 条件）的梯度计算"""

        def f(a):
            """复杂函数：循环倍增直到范数 >=1000，然后条件赋值"""
            b = a * 2
            while b.norm() < 1000:  # 动态循环，次数取决于 |b|，即 b 的范数，标量的范数为其绝对值
                b = b * 2
            if b.sum() > 0:  # 条件分支（标量sum() == b），标量的 sum() 是其自身
                c = b
            else:
                c = 100 * b
            return c

        # 创建随机标量 a，如 0.4516275227069855
        a = torch.randn(size=(), requires_grad=True)
        print(f"Input a: {a}")

        # 前向计算 d，如 1849.8663330078125
        d = f(a)
        print(f"Output d: {d}")

        # 反向传播
        d.backward()
        print(f"a.grad: {a.grad}")

        # 验证：由于 f(a) = k * a（k由路径决定），梯度应为 d / a
        if a != 0:  # 避免 division by zero（极小概率）
            expected_grad = d / a
            print(f"Expected grad (d / a): {expected_grad}")
            self.assertTrue(torch.allclose(a.grad, expected_grad))
        else:
            self.assertTrue(torch.isnan(a.grad))  # 如果 a=0，梯度应为 NaN 或 0，根据上下文

    def test_higher_order_grad(self):
        """演示一阶 vs 二阶导数的计算开销"""
        x = torch.tensor(2.0, requires_grad=True)

        # 一阶导数：y = x^2，dy/dx = 2x
        start_time = time.time()
        y = x ** 2
        y.backward()
        grad1 = x.grad
        print(f"First-order grad: {grad1}, time: {time.time() - start_time:.6f}s")
        self.assertTrue(torch.allclose(grad1, 2 * x))

        x.grad.zero_()  # 清除

        # 二阶导数：d²y/dx² = 2（使用 autograd.grad）
        y = x ** 2
        grad1 = torch.autograd.grad(y, x, create_graph=True)[0]  # 一阶
        start_time = time.time()
        grad2 = torch.autograd.grad(grad1, x)[0]  # 二阶
        print(f"Second-order grad: {grad2}, time: {time.time() - start_time:.6f}s")
        self.assertTrue(torch.allclose(grad2, torch.tensor(2.0)))

        # 注意：二阶时间/内存通常更高；在更大模型中差异更明显

    def test_backward_twice(self):
        """测试多次调用 backward() 的行为"""
        x = torch.tensor(1.0, requires_grad=True)
        y = x ** 2

        # 第一次 backward：成功
        y.backward()
        print(f"First backward: x.grad = {x.grad}")
        self.assertTrue(torch.allclose(x.grad, torch.tensor(2.0)))

        # 第二次：报错（计算图已销毁）
        try:
            y.backward()
            self.fail("Expected error on second backward")
        except RuntimeError as e:
            print(f"Error on second backward: {str(e)}")

        # 使用 retain_graph=True 允许多次（梯度累加）
        x.grad.zero_()
        y = x ** 2
        y.backward(retain_graph=True)
        print(f"With retain_graph=True: After first = {x.grad}", end="")
        y.backward()  # 现在成功，累加
        print(f", after second = {x.grad}")
        self.assertTrue(torch.allclose(x.grad, torch.tensor(4.0)))  # 2 + 2

    def test_control_flow_vector(self):
        """测试控制流例子中 a 改为向量"""

        def f(a):
            b = a * 2
            while b.norm() < 1000:  # 整体 norm（向量 L2 范数）
                b = b * 2
            if b.sum() > 0:  # 整体 sum
                c = b
            else:
                c = 100 * b
            return c

        # a 改为随机向量
        a = torch.randn(2, requires_grad=True)
        print(f"Vector a: {a}")

        d = f(a)
        print(f"Output d: {d}")

        d.sum().backward()  # 因为 d 是向量，求 sum 的梯度
        print(f"a.grad: {a.grad}")

        # 验证：仍为 d / a（逐元素）
        expected_grad = d / a
        print(f"Expected grad (d / a): {expected_grad}")
        self.assertTrue(torch.allclose(a.grad, expected_grad, equal_nan=True))

    def test_custom_control_flow_grad(self):
        """自定义控制流梯度例子：循环 + 逐元素条件"""

        def g(x, num_loops=2):
            y = torch.zeros_like(x)
            for _ in range(num_loops):  # 固定循环（动态也可）
                # 逐元素条件：>0 时累加 x^2，否则累加 x*2
                y += torch.where(x > 0, x ** 2, x * 2)
            return y

        x = torch.tensor([-1.0, 1.0], requires_grad=True)
        print(f"Input x: {x}")

        y = g(x)
        print(f"Output y after control flow: {y}")

        total = y.sum()
        print(f"Sum y: {total}")
        total.backward()
        print(f"x.grad: {x.grad}")

        # 预期梯度：负元素每个循环导数=2（累加），正元素=2x（累加）
        expected_grad = torch.tensor([2.0 * 2, 2.0 * 1 * 2])  # 2 loops
        print(f"Expected grad: {expected_grad}")
        self.assertTrue(torch.allclose(x.grad, expected_grad))

        # 分析：autograd 正确跟踪逐元素路径和循环累加


if __name__ == '__main__':
    unittest.main()
