from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


def load_fashion_mnist():
    print("Loading Fashion-MNIST...")

    dataset = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=False
    )

    X = dataset.data.astype(np.uint8)
    y = dataset.target.astype(np.int64)

    print(f"Images shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y


def show_sample(X, y, index=0):
    image = X[index].reshape(28, 28)
    label = y[index]

    plt.imshow(image, cmap="gray")
    plt.title(f"Label: {label} - {CLASS_NAMES[label]}")
    plt.axis("off")
    plt.show()
def show_samples(X, y, count=10):
    plt.figure(figsize=(15, 4))

    for i in range(count):
        image = X[i].reshape(28, 28)

        plt.subplot(2, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(CLASS_NAMES[y[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
def show_class_distribution(y):
    counts = Counter(y)

    labels = list(range(10))
    values = [counts[label] for label in labels]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)

    plt.xticks(labels, CLASS_NAMES, rotation=45)
    plt.ylabel("Number of Images")
    plt.xlabel("Class")
    plt.title("Fashion-MNIST Class Distribution")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_fashion_mnist()

    show_samples(X, y)
    show_class_distribution(y)
