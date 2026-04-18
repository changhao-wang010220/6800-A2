import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def balanced_accuracy(y_true, y_pred, num_classes=10):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    recalls = []
    for class_id in range(num_classes):
        total = matrix[class_id].sum()
        if total == 0:
            continue
        recalls.append(matrix[class_id, class_id] / total)
    return float(np.mean(recalls))


def macro_f1(y_true, y_pred, num_classes=10):
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    scores = []

    for class_id in range(num_classes):
        tp = matrix[class_id, class_id]
        fp = matrix[:, class_id].sum() - tp
        fn = matrix[class_id, :].sum() - tp

        denominator = 2 * tp + fp + fn
        if denominator == 0:
            scores.append(0.0)
        else:
            scores.append((2 * tp) / denominator)

    return float(np.mean(scores))


def cross_entropy(y_true, y_prob, eps=1e-15):
    """
    y_prob should be model probabilities with shape (num_samples, 10).
    """

    y_prob = np.clip(y_prob, eps, 1 - eps)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    return float(-np.log(y_prob[np.arange(len(y_true)), y_true]).mean())


def print_metrics(y_true, y_pred, y_prob=None):
    print(f"Accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy(y_true, y_pred):.4f}")
    print(f"Macro F1-score: {macro_f1(y_true, y_pred):.4f}")

    if y_prob is not None:
        print(f"Cross entropy: {cross_entropy(y_true, y_prob):.4f}")
