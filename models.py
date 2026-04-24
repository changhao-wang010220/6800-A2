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
    CNN(卷积神经网络)。

    输入形状:
        (batch_size, 1, 28, 28)

    输出形状:
        (batch_size, 10)
    """

    def __init__(self):
        if nn is None:
            raise ImportError("Install torch before implementing the model.")
        super().__init__()

        # 定义特征提取部分，按顺序堆叠多层网络
        # 第一层卷积：输入1个通道，输出16个特征图，卷积核3x3，padding=1 保持尺寸不变
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            # 激活函数，增加非线性能力
            nn.ReLU(),
            # 最大池化层：用2x2窗口把特征图长宽减半（28x28 ->14x14）
            nn.MaxPool2d(kernel_size=2),
            # 第二层卷积：输入16个通道，输出32个特征图
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),)

        # 定义分类部分，把卷积提取到的特征送去分类
        self.classifier = nn.Sequential(
            # 把四维张量展平成二维张量，方便输入全连接层
            nn.Flatten(),
            # 第一层全连接：输入32*7*7个特征，输出128个神经元
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # Dropout：训练时随机丢弃 30% 神经元，减少过拟合
            nn.Dropout(p=0.30),
            # 输出层：128维映射到10类
            nn.Linear(128, 10),)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    """
    CNN设计
    在具体结构上，模型输入为形状为(1,28,28)的单通道灰度图像。第一层卷积层将通道数从1提升到16，卷积核大小设为3x3，
    并采用padding=1，从而在提取局部边缘和纹理特征的同时保持特征图的空间尺寸不变。经过ReLU激活函数后，使用一次2x2
    最大池化，将特征图尺寸从28x28降为14x14，以减少参数规模并保留更显著的局部信息。第二层卷积层进一步将通道数从16
    提升到32，同样使用3x3卷积核和padding=1，以学习更高层次的笔画组合与形状特征；之后再次经过ReLU和最大池化，使特
    征图尺寸进一步缩小为7x7。

    在卷积特征提取完成后，模型将得到的32x7x7特征图展平为一维向量，并送入全连接分类器。分类器的第一层全连接层将1568
    个输入特征映射到128个隐藏神经元，随后使用ReLU激活函数增强非线性表达能力，并加入丢弃率为0.30的Dropout层，以缓解
    过拟合问题。最后一层全连接层将128维隐藏表示映射到10个输出节点，对应数字0到9的十个类别。模型最后输出的是logits，
    在测试阶段通过softmax将logits转换为各类别概率，用于计算交叉熵等指标。
    """
