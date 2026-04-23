try:
    import torch.nn as nn
except ImportError:
    nn = None


class FullyConnectedNet(nn.Module if nn else object):
    """
    模型: 全连接神经网络，用于 MNIST 手写数字分类

    输入形状: (batch_size, 784)
    MNIST 每张图是 28 × 28，摊平成一维变成 28 × 28 = 784 个数字
        

    输出形状: (batch_size, 10)
    数字分成 0~9 共 10 类，所以模型最后会输出 10 个值，分别代表每一类的分数
    """

    def __init__(self):
        if nn is None:
            raise ImportError("Install torch before implementing the model.")
        super().__init__()

        # 网络结构设计

        # 层数变化：784 → 256 → 128 → 10
        # 输入有 784 个特徵
        # 第一层有 256 个神经元
        # 第二层有 128 个神经元
        # 输出有 10 个节点

        # 每一层后使用 ReLU 激活函数 (帮模型加入非线性能力，强化模型)
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        # 如果输入是图像 (batch, 1, 28, 28)，需要 flatten
        # 检查输入 x 的维度数，如果大于 2，代表它还是图像形式，还没 flatten
        if x.dim() > 2:
            # 做 flatten
            x = x.view(x.size(0), -1)

        return self.network(x)
    

    """
    全连接神经网络设计

    在这个模型中，我们先把 MNIST 的图像做预处理。因为原始图片大小是 28×28，
    所以在输入全连接神经网络之前，会先把每张图片展平成一个长度为 784 的一维向量。
    这样做的好处是可以让模型把每一个像素当作一个输入特征，再透过后面的线性层去学习
    这些像素之间的组合关系。

    在网络结构上，我们设计了一个简单的三层结构。首先是输入层 784，
    接着是两个隐藏层，分别是 256 和 128 个神经元，每一层后面都接 ReLU
    激活函数，用来增加模型的非线性能力，让模型可以学习比较复杂的模式。
    最后一层是输出层，有 10 个节点，对应数字 0 到 9 这 10 个类别。

    模型的输出是 logits，也就是还没有经过 softmax 的分数。在训练的时候，
    我们会直接把 logits 丢进交叉熵损失函数来做优化；在测试的时候，
    再用 softmax 把它转成概率，用来判断预测结果。
    """


class CNN(nn.Module if nn else object):
    """
    TODO: 这里由负责 CNN 的组员补全。

    输入形状:
        (batch_size, 1, 28, 28)

    输出形状:
        (batch_size, 10)
    """

    def __init__(self):
        if nn is None:
            raise ImportError("Install torch before implementing the model.")
        super().__init__()

        # TODO: 在这里定义 CNN 的卷积层、池化层和全连接层。
        raise NotImplementedError("CNN still needs to be implemented.")

    def forward(self, x):
        # TODO: 在这里写前向传播。
        raise NotImplementedError("CNN.forward still needs to be implemented.")