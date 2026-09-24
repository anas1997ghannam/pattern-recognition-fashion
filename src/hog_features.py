import numpy as np
from skimage.feature import hog


def extract_hog_features(X):
    """
    Extract HOG features from Fashion-MNIST images.

    Parameters
    ----------
    X : numpy.ndarray
        Images represented as flattened 784-dimensional vectors.

    Returns
    -------
    numpy.ndarray
        HOG feature matrix.
    """

    # Convert 784-dimensional vectors back to 28x28 images
    images = X.reshape(-1, 28, 28)

    hog_features = []

    for image in images:
        features = hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )

        hog_features.append(features)

    return np.array(hog_features)


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    print("Loading Fashion-MNIST...")

    dataset = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=False,
    )

    X = dataset.data.astype(np.float32)

    # Normalize pixel values
    X = X / 255.0

    # Use a small subset for the first test
    X_sample = X[:1000]

    print(f"Input shape: {X_sample.shape}")

    print("Extracting HOG features...")

    X_hog = extract_hog_features(X_sample)

    print(f"HOG feature shape: {X_hog.shape}")


"""
number of hog features=1296 because each image is 28x28 pixels, and with the specified parameters (orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)), the resulting HOG feature vector has a length of 1296.

python src/hog_features.py
Loading Fashion-MNIST...
Input shape: (1000, 784)
Extracting HOG features...
HOG feature shape: (1000, 1296)
"""