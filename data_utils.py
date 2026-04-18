import struct
from pathlib import Path

import numpy as np


DATA_DIR = Path("MNIST")

TRAIN_IMAGES = DATA_DIR / "train-images.idx3-ubyte"
TRAIN_LABELS = DATA_DIR / "train-labels.idx1-ubyte"
TEST_IMAGES = DATA_DIR / "t10k-images.idx3-ubyte"
TEST_LABELS = DATA_DIR / "t10k-labels.idx1-ubyte"


class MnistDataloader:
    """
    MNIST loader based on the read-mnist-dataset notebook example.

    The notebook returns Python lists. This version returns numpy arrays because
    they are easier to pass into PyTorch/TensorFlow training code.
    """

    def __init__(
        self,
        training_images_filepath=TRAIN_IMAGES,
        training_labels_filepath=TRAIN_LABELS,
        test_images_filepath=TEST_IMAGES,
        test_labels_filepath=TEST_LABELS,
        normalize=True,
    ):
        self.training_images_filepath = Path(training_images_filepath)
        self.training_labels_filepath = Path(training_labels_filepath)
        self.test_images_filepath = Path(test_images_filepath)
        self.test_labels_filepath = Path(test_labels_filepath)
        self.normalize = normalize

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = self.read_labels(labels_filepath)
        images = self.read_images(images_filepath)
        return images, labels

    def read_labels(self, labels_filepath):
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    f"Magic number mismatch, expected 2049, got {magic}"
                )

            labels = np.frombuffer(file.read(), dtype=np.uint8)

        if labels.shape[0] != size:
            raise ValueError(
                f"Label count mismatch, expected {size}, got {labels.shape[0]}"
            )

        return labels.astype(np.int64)

    def read_images(self, images_filepath):
        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f"Magic number mismatch, expected 2051, got {magic}"
                )

            image_data = np.frombuffer(file.read(), dtype=np.uint8)

        images = image_data.reshape(size, rows, cols)

        if self.normalize:
            images = images.astype(np.float32) / 255.0

        return images

    def load_data(self):
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath,
            self.training_labels_filepath,
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath,
            self.test_labels_filepath,
        )
        return (x_train, y_train), (x_test, y_test)


def load_mnist(data_dir=DATA_DIR, normalize=True):
    """
    Simple helper used by train.py.

    Returns:
        x_train, y_train, x_test, y_test
    """

    data_dir = Path(data_dir)
    loader = MnistDataloader(
        training_images_filepath=data_dir / "train-images.idx3-ubyte",
        training_labels_filepath=data_dir / "train-labels.idx1-ubyte",
        test_images_filepath=data_dir / "t10k-images.idx3-ubyte",
        test_labels_filepath=data_dir / "t10k-labels.idx1-ubyte",
        normalize=normalize,
    )

    (x_train, y_train), (x_test, y_test) = loader.load_data()
    return x_train, y_train, x_test, y_test


def prepare_for_fcnn(images):
    """Convert images from (N, 28, 28) to (N, 784)."""

    return images.reshape(images.shape[0], -1)


def prepare_for_cnn(images):
    """Convert images from (N, 28, 28) to PyTorch CNN shape (N, 1, 28, 28)."""

    return images[:, None, :, :]
