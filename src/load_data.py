# imports a scikit-learn tool to load datasets available on OpenML
from sklearn.datasets import fetch_openml
import numpy as np


def load_fashion_mnist():
    print("Loading Fashion-MNIST...")
    # Fashion-MNIST version 1
    dataset = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=False #numpy arrays not pandas DataFrame
    )

    X = dataset.data #input data each row is a 28x28 image flattened to a 784-dimensional vector
    y = dataset.target.astype(np.int64)# target labels (0-9) for each image, representing different clothing categories 0 T-shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot

    X = X.astype(np.uint8)# reduce memory usage by converting to uint8 (0-255 pixel values), because Fashion-MNIST images are grayscale images with pixel values ranging from 0 to 255

    print(f"Images shape: {X.shape}")# (70000, 784) because there are 70,000 images in the dataset, each represented as a 784-dimensional vector (28x28 pixels flattened)
    print(f"Labels shape: {y.shape}")

    return X, y


if __name__ == "__main__":
    X, y = load_fashion_mnist()
