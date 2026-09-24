
import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from hog_features import extract_hog_features


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
    "Ankle boot",
]


def load_fashion_mnist():
    print("Loading Fashion-MNIST...")

    dataset = fetch_openml(
        name="Fashion-MNIST",
        version=1,
        as_frame=False,
    )

    X = dataset.data.astype(np.float32)
    y = dataset.target.astype(np.int64)

    return X, y


def calculate_confusion_values(cm, class_index):
    """
    Calculate TP, TN, FP and FN for one class
    using a one-vs-rest approach.
    """

    tp = cm[class_index, class_index]

    fn = cm[class_index, :].sum() - tp

    fp = cm[:, class_index].sum() - tp

    tn = cm.sum() - (tp + fn + fp)

    return tp, tn, fp, fn


def main():
    # ============================================================
    # 1. Load Dataset
    # ============================================================

    X, y = load_fashion_mnist()

    # ============================================================
    # 2. Train / Validation / Test Split
    # ============================================================

    X_train = X[:50000]
    y_train = y[:50000]

    X_validation = X[50000:60000]
    y_validation = y[50000:60000]

    X_test = X[60000:]
    y_test = y[60000:]

    print(f"Training samples: {X_train.shape}")
    print(f"Validation samples: {X_validation.shape}")
    print(f"Testing samples: {X_test.shape}")

    # ============================================================
    # 3. Normalization
    # ============================================================

    X_train = X_train / 255.0
    X_validation = X_validation / 255.0
    X_test = X_test / 255.0

    # ============================================================
    # 4. HOG Feature Extraction
    # ============================================================

    print("\nExtracting HOG features...")

    start_time = time.time()

    X_train_hog = extract_hog_features(X_train)
    X_validation_hog = extract_hog_features(X_validation)
    X_test_hog = extract_hog_features(X_test)

    hog_time = time.time() - start_time

    print(f"HOG extraction time: {hog_time:.2f} seconds")

    print(f"HOG training shape: {X_train_hog.shape}")
    print(f"HOG validation shape: {X_validation_hog.shape}")
    print(f"HOG testing shape: {X_test_hog.shape}")

    # ============================================================
    # 5. Train Linear SVM
    # ============================================================

    print("\nTraining Linear SVM with HOG features...")

    model = LinearSVC(
        C=1.0,
        max_iter=3000,
        random_state=42,
    )

    start_time = time.time()

    model.fit(X_train_hog, y_train)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    # ============================================================
    # 6. Validation
    # ============================================================

    print("\nEvaluating on validation set...")

    y_validation_pred = model.predict(X_validation_hog)

    validation_accuracy = accuracy_score(
        y_validation,
        y_validation_pred,
    )

    print(f"Validation Accuracy: {validation_accuracy:.4f}")

    # ============================================================
    # 7. Final Test Evaluation
    # ============================================================

    print("\nEvaluating on test set...")

    start_time = time.time()

    y_test_pred = model.predict(X_test_hog)

    prediction_time = time.time() - start_time

    test_accuracy = accuracy_score(
        y_test,
        y_test_pred,
    )

    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # ============================================================
    # 8. Classification Report
    # ============================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=CLASS_NAMES,
        )
    )

    # ============================================================
    # 9. Confusion Matrix
    # ============================================================

    cm = confusion_matrix(
        y_test,
        y_test_pred,
    )

    print("\nConfusion Matrix:")
    print(cm)

    # ============================================================
    # 10. TP / TN / FP / FN
    # ============================================================

    print("\nPer-class TP / TN / FP / FN:")

    for class_index, class_name in enumerate(CLASS_NAMES):

        tp, tn, fp, fn = calculate_confusion_values(
            cm,
            class_index,
        )

        print(
            f"{class_name:15s} "
            f"TP={tp:5d} "
            f"TN={tn:5d} "
            f"FP={fp:5d} "
            f"FN={fn:5d}"
        )


if __name__ == "__main__":
    main()
