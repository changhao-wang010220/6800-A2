import numpy as np

from data_utils import load_mnist, prepare_for_cnn, prepare_for_fcnn
from metrics import print_metrics


RANDOM_SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


def train_fully_connected_network(x_train, y_train, x_test, y_test):
    """
    训练并评估全连接神经网络（Fully Connected Neural Network）

    1. 数据预处理（flatten）
    2. 建立模型
    3. 模型训练
    4. 测试集预测
    5. 输出评估指标
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from models import FullyConnectedNet

    # 固定随机种子
    torch.manual_seed(RANDOM_SEED)

    # 把 28x28 图像展平成 784 维向量
    x_train = prepare_for_fcnn(x_train)
    x_test = prepare_for_fcnn(x_test)

    print("全连接神经网络部分")
    print(f"flatten 后的 x_train shape: {x_train.shape}")
    print(f"flatten 后的 x_test shape: {x_test.shape}")

    # 转换成 PyTorch tensor
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 建立训练集和测试集
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # DataLoader 用来分 batch 读取数据
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 选择运算装置：有 GPU 就用 GPU，没有就用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立模型并移动到 device
    model = FullyConnectedNet().to(device)

    # 多分类任务常用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # Adam 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # =============================
    # 模型训练
    # =============================
    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # 每次更新前先把梯度清空
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_images)

            # 计算 loss
            loss = criterion(outputs, batch_labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计这一轮的 loss 和 accuracy
            running_loss += loss.item() * batch_images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    # =============================
    # 模型测试
    # =============================
    model.eval()

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for batch_images, _ in test_loader:
            batch_images = batch_images.to(device)

            outputs = model(batch_images)

            # 把 logits 转成概率
            probs = torch.softmax(outputs, dim=1)

            # 取得预测类别
            preds = outputs.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())

    # 拼接所有 batch 的结果
    y_prob = torch.cat(all_probs, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    print("测试集评估结果:")
    print_metrics(y_test_tensor.numpy(), y_pred, y_prob)


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
