import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
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
    训练并评估CNN。
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from models import CNN
    # 设置随机种子，保证结果尽量可复现
    torch.manual_seed(RANDOM_SEED)

    x_train = prepare_for_cnn(x_train)
    x_test = prepare_for_cnn(x_test)

    print("CNN 部分")
    print(f"增加 channel 后的 x_train shape: {x_train.shape}")
    print(f"增加 channel 后的 x_test shape: {x_test.shape}")

    # 转换训练测试图像与标签类型
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    # 创建训练测试数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 定义Adam优化器，用于根据梯度更新模型参数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 模型训练
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # 按批次读取训练数据并更新模型参数
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            # 统计当前轮训练损失与准确率
            running_loss += loss.item() * batch_images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
        # 输出每一轮训练结果
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

    # 模型测试
    model.eval()
    all_probs = []
    all_preds = []
    # 在测试集上进行预测
    with torch.no_grad():
        for batch_images, _ in test_loader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
    # 整理结果并输出指标
    y_prob = torch.cat(all_probs, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    print_metrics(y_test_tensor.numpy(), y_pred, y_prob)
    

    # Captum 开始
    # 生成一个 Captum 模型解释结果
    explain_cnn_with_captum(
        model=model,
        x_test_tensor=x_test_tensor,
        y_test_tensor=y_test_tensor,
        device=device,
        sample_index=0,
        save_path="captum_mnist_explanation.png")

def explain_cnn_with_captum(model, x_test_tensor, y_test_tensor, device,
                            sample_index=0,
                            save_path="captum_mnist_explanation.png"):
    """
    使用 Captum 的 Integrated Gradients 对 CNN 的单个测试样本进行解释，
    并保存解释图像。
    """
    import torch

    model.eval()

    # 取一个测试样本
    input_image = x_test_tensor[sample_index:sample_index + 1].to(device)
    true_label = int(y_test_tensor[sample_index].item())

    # 前向预测
    input_image.requires_grad_()
    outputs = model(input_image)
    pred_label = int(outputs.argmax(dim=1).item())

    # 定义 Captum 解释器
    ig = IntegratedGradients(model)

    baseline = torch.zeros_like(input_image)

    # 针对预测类别做归因
    attributions, delta = ig.attribute(
        input_image,
        baselines=baseline,
        target=pred_label,
        return_convergence_delta=True
    )

    image_np = input_image.detach().cpu().squeeze().numpy()
    attr_np = attributions.detach().cpu().squeeze().numpy()

    # 为了可视化，取绝对值并归一化
    attr_vis = np.abs(attr_np)
    attr_vis = attr_vis / (attr_vis.max() + 1e-8)

    # 绘图
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title(f"Original\nTrue={true_label}, Pred={pred_label}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attr_np, cmap="seismic")
    plt.title("IG Attribution")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_np, cmap="gray")
    plt.imshow(attr_vis, cmap="hot", alpha=0.5)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Captum explanation saved to: {save_path}")
    print(f"Convergence delta: {delta.item():.6f}")
    # Captum 结束

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
