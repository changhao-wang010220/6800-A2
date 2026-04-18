import numpy as np

from data_utils import load_mnist, prepare_for_cnn, prepare_for_fcnn
from metrics import print_metrics


RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


def train_fully_connected_network(x_train, y_train, x_test, y_test):
    """
    TODO: 这里由负责全连接神经网络的组员补全训练和测试流程。
    """

    x_train = prepare_for_fcnn(x_train)
    x_test = prepare_for_fcnn(x_test)

    print("全连接神经网络部分")
    print(f"flatten 后的 x_train shape: {x_train.shape}")
    print(f"flatten 后的 x_test shape: {x_test.shape}")
    print("TODO: 在这里补全全连接神经网络的训练代码。")

    # 临时 baseline：保证模型还没写时，框架也能先跑起来。
    majority_label = int(np.bincount(y_train).argmax())
    y_pred = np.full_like(y_test, majority_label)
    print_metrics(y_test, y_pred)


def train_cnn(x_train, y_train, x_test, y_test):
    """
    TODO: 这里由负责 CNN 的组员补全训练和测试流程。
    """

    x_train = prepare_for_cnn(x_train)
    x_test = prepare_for_cnn(x_test)

    print("CNN 部分")
    print(f"增加 channel 后的 x_train shape: {x_train.shape}")
    print(f"增加 channel 后的 x_test shape: {x_test.shape}")
    print("TODO: 在这里补全 CNN 的训练代码。")

    # 临时 baseline：保证模型还没写时，框架也能先跑起来。
    majority_label = int(np.bincount(y_train).argmax())
    y_pred = np.full_like(y_test, majority_label)
    print_metrics(y_test, y_pred)


def main():
    np.random.seed(RANDOM_SEED)

    x_train, y_train, x_test, y_test = load_mnist()

    print("MNIST 数据读取成功。")
    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")
    print()

    train_fully_connected_network(x_train, y_train, x_test, y_test)
    print()
    train_cnn(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
    main()
