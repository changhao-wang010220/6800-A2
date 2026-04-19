try:
    import torch.nn as nn
except ImportError:
    nn = None


class FullyConnectedNet(nn.Module if nn else object):
    """
    TODO: 这里由负责全连接神经网络的组员补全。

    输入形状:
        (batch_size, 784)

    输出形状:
        (batch_size, 10)
    """

    def __init__(self):
        if nn is None:
            raise ImportError("Install torch before implementing the model.")
        super().__init__()

        # TODO: 在这里定义全连接网络的层。
        raise NotImplementedError("FullyConnectedNet still needs to be implemented.")

    def forward(self, x):
        # TODO: 在这里写前向传播。
        raise NotImplementedError("FullyConnectedNet.forward still needs to be implemented.")


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
            # 第二层卷积：输入6个通道，输出32个特征图
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
