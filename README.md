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

## 全连接神经网络 TODO

在 `models.py` 里：

- 补全 `FullyConnectedNet`

在 `train.py` 里：

- 补全 `train_fully_connected_network`
- 把 numpy 数据转成 tensor
- 创建 DataLoader
- 训练模型
- 用 `print_metrics` 输出评估指标

## CNN TODO

在 `models.py` 里：

- 补全 `CNN`

在 `train.py` 里：

- 补全 `train_cnn`
- 把 numpy 数据转成 tensor
- 创建 DataLoader
- 训练模型
- 用 `print_metrics` 输出评估指标

## 评估指标

- Accuracy
- Balanced accuracy
- Macro F1-score
- Cross entropy，如果模型能输出每个类别的概率

## 数据说明
当前代码使用的数据文件是这四个：

```text
MNIST/train-images.idx3-ubyte
MNIST/train-labels.idx1-ubyte
MNIST/t10k-images.idx3-ubyte
MNIST/t10k-labels.idx1-ubyte
```
