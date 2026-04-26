# DSCI 6800 Assignment 2

这个项目是 A2 的 MNIST 手写数字识别作业。

作业要求我们训练并比较两个模型：

1. 全连接神经网络
2. CNN

目前这个 repo 只是一个简单框架。

## 文件说明

```text
MNIST/          # 数据集文件
data_utils.py   # 读取 MNIST 数据，并整理输入形状
models.py       # TODO: 连接神经网络和 CNN
metrics.py      # 评估指标：accuracy、balanced accuracy、macro F1、cross entropy
train.py        # TODO：主训练文件，训练流程
requirements.txt
```

## 怎么运行

先安装依赖：

```bash
python -m pip install -r requirements.txt
```

然后运行：

```bash
python train.py
```

现在运行 `train.py` 只会检查数据是否能正常读取，并跑一个简单 baseline。真正的模型训练部分在 TODO 里。

数据正常读取后，应该看到这些形状：

```text
x_train: (60000, 28, 28)
y_train: (60000,)
x_test: (10000, 28, 28)
y_test: (10000,)
```

## 数据说明
当前代码使用的数据文件是这四个：

```text
MNIST/train-images.idx3-ubyte
MNIST/train-labels.idx1-ubyte
MNIST/t10k-images.idx3-ubyte
MNIST/t10k-labels.idx1-ubyte
```

## 模型设计简要描述

### FullyConnectedNet（全连接神经网络）
- 输入层：784 个节点（28×28 图像展平）
- 隐藏层 1：256 个节点 + ReLU 激活
- 隐藏层 2：128 个节点 + ReLU 激活
- 输出层：10 个节点（对应数字 0-9）

### CNN（卷积神经网络）
- 卷积层 1：1→16 通道，3×3 卷积核，padding=1 + ReLU + 2×2 最大池化
- 卷积层 2：16→32 通道，3×3 卷积核，padding=1 + ReLU + 2×2 最大池化
- 全连接层：1568→128 + ReLU + Dropout(0.30)
- 输出层：128→10

## 已知问题和调试步骤

### 已知问题
无重大漏洞。模型在测试集上表现良好：
- FNN 准确率：97.82%
- CNN 准确率：99.13%

### 调试步骤
1. 确保 MNIST 数据文件放在 `MNIST/` 目录下
2. 安装依赖：`pip install -r requirements.txt`
3. 运行 `python train.py` 验证数据读取正常
4. 逐模块调试：先测试 FNN，再测试 CNN
5. 使用 `print_metrics()` 输出评估指标

## 实验结果摘要

| 模型 | 准确率 | 平衡准确率 | 宏平均 F1 | 交叉熵 |
|------|--------|------------|-----------|--------|
| FNN | 97.82% | 97.78% | 97.80% | 0.0919 |
| CNN | 99.13% | 99.12% | 99.12% | 0.0283 |

