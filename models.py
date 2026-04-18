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
