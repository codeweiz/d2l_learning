import unittest
import os
import tempfile
import shutil
import pandas as pd
import torch
import numpy as np


class TestPandas(unittest.TestCase):

    def setUp(self):
        # 创建临时目录和 CSV 文件
        self.temp_dir = tempfile.mkdtemp()
        self.data_file = os.path.join(self.temp_dir, 'house_tiny.csv')
        with open(self.data_file, 'w') as f:
            f.write('NumRooms,Alley,Price\n')
            f.write('NA,Pave,127500\n')
            f.write('2,NA,106000\n')
            f.write('4,NA,178100\n')
            f.write('NA,NA,140000\n')

    def tearDown(self):
        # 清理临时目录
        shutil.rmtree(self.temp_dir)

    # 测试读取数据
    def test_read_data(self):
        data = pd.read_csv(self.data_file)
        print(data)  # 可选打印

        # 断言：检查形状、列名和具体值
        self.assertEqual(data.shape, (4, 3), "DataFrame 形状应为 (4, 3)")
        self.assertListEqual(list(data.columns), ['NumRooms', 'Alley', 'Price'], "列名不匹配")
        self.assertTrue(np.isnan(data['NumRooms'].iloc[0]), "第一行 NumRooms 应为 NaN")
        self.assertEqual(data['Price'].iloc[0], 127500, "第一行 Price 值不匹配")

    # 测试处理缺失值
    def test_handle_default(self):
        data = pd.read_csv(self.data_file)
        inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

        # 用每列的平均值填充 NaN（仅数值列）
        inputs = inputs.fillna(inputs.mean(numeric_only=True))
        print(inputs)  # 可选打印

        # 使用 get_dummies 转换为哑变量
        inputs = pd.get_dummies(inputs, dummy_na=True)
        print(inputs)  # 可选打印

        # 断言：检查填充后缺失值、形状和具体值
        self.assertFalse(inputs.isnull().values.any(), "填充后不应有 NaN")
        self.assertEqual(inputs.shape, (4, 3),
                         "get_dummies 后形状应为 (4, 3)")  # NumRooms + Alley_Pave + Alley_nan + Alley_其他（但数据中只有 Pave 和 nan）
        self.assertEqual(inputs['NumRooms'].mean(), 3.0, "NumRooms 的平均值应为 3.0")  # (2 + 4) / 2 = 3

    # 测试转换为张量格式
    def test_convert_to_tensor(self):
        data = pd.read_csv(self.data_file)
        inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
        inputs = inputs.fillna(inputs.mean(numeric_only=True))
        inputs = pd.get_dummies(inputs, dummy_na=True)

        # 转换为张量
        X = torch.tensor(inputs.to_numpy(dtype=float))
        y = torch.tensor(outputs.to_numpy(dtype=float))
        print(X, y)  # 可选打印

        # 断言：检查类型、形状和值
        self.assertIsInstance(X, torch.Tensor, "X 应为 torch.Tensor")
        self.assertIsInstance(y, torch.Tensor, "y 应为 torch.Tensor")
        self.assertEqual(X.shape, torch.Size([4, 3]), "X 形状应为 [4, 3]")
        self.assertEqual(y.shape, torch.Size([4]), "y 形状应为 [4]")
        self.assertTrue(torch.allclose(X[0, 0].float(), torch.tensor(3.0).float()), "X[0, 0] 应为 3.0（填充值）")
        self.assertEqual(y[0].item(), 127500.0, "y[0] 值不匹配")

    # 测试删除缺失值最多的列，并将预处理后的数据集转换为张量格式
    def test(self):
        data = pd.read_csv(self.data_file)

        # 计算每列的缺失值数量
        missing_counts = data.isnull().sum()
        # 找出缺失值最多的列（如果有多个，选择第一个）
        max_missing_col = missing_counts.idxmax()
        # 删除该列
        data = data.drop(columns=[max_missing_col])
        print(f"Deleted column with most missing values: {max_missing_col}")

        # 剩余处理
        inputs, outputs = data.iloc[:, 0:1], data.iloc[:, 1]  # 假设删除后列调整
        inputs = inputs.fillna(inputs.mean(numeric_only=True))
        inputs = pd.get_dummies(inputs, dummy_na=True)

        # 转换为张量
        X = torch.tensor(inputs.to_numpy(dtype=float))
        y = torch.tensor(outputs.to_numpy(dtype=float))
        print(X, y)  # 可选打印

        # 断言：检查删除的列、形状和值
        self.assertNotIn(max_missing_col, data.columns, f"列 {max_missing_col} 未被删除")
        self.assertEqual(data.shape, (4, 2), "删除后 DataFrame 形状应为 (4, 2)")  # 原 (4,3) 删除一列
        self.assertEqual(inputs.shape[1], 1, "inputs 列数应为 1（假设删除 Alley 或 NumRooms）")  # 取决于哪列缺失最多
        self.assertFalse(inputs.isnull().values.any(), "填充后不应有 NaN")
        self.assertIsInstance(X, torch.Tensor, "X 应为 torch.Tensor")
        self.assertEqual(X.shape[0], 4, "X 行数应为 4")


if __name__ == '__main__':
    unittest.main()
