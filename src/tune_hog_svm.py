import time

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from hog_features import extract_hog_features


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


def main():
    # ============================================================
    # 1. Load Dataset
    # ============================================================

    X, y = load_fashion_mnist()

    # ============================================================
    # 2. Train / Validation Split
    # ============================================================

    X_train = X[:50000]
    y_train = y[:50000]

    X_validation = X[50000:60000]
    y_validation = y[50000:60000]

    # ============================================================
    # 3. Normalization
    # ============================================================

    X_train = X_train / 255.0
    X_validation = X_validation / 255.0

    # ============================================================
    # 4. HOG Feature Extraction
    # ============================================================

    print("\nExtracting HOG features...")

    start_time = time.time()

    X_train_hog = extract_hog_features(X_train)
    X_validation_hog = extract_hog_features(X_validation)

    hog_time = time.time() - start_time

    print(f"HOG extraction time: {hog_time:.2f} seconds")
    print(f"Training HOG shape: {X_train_hog.shape}")
    print(f"Validation HOG shape: {X_validation_hog.shape}")

    # ============================================================
    # 5. Hyperparameter Search
    # ============================================================

    C_values = [0.01, 0.1, 1.0, 10.0]

    results = []

    print("\nStarting C-value search...")

    for C in C_values:

        print(f"\nTraining Linear SVM with C={C}")

        model = LinearSVC(
            C=C,
            max_iter=3000,
            random_state=42,
        )

        start_time = time.time()

        model.fit(
            X_train_hog,
            y_train,
        )

        training_time = time.time() - start_time

        y_validation_pred = model.predict(
            X_validation_hog
        )

        validation_accuracy = accuracy_score(
            y_validation,
            y_validation_pred,
        )

        results.append(
            {
                "C": C,
                "Validation Accuracy": validation_accuracy,
                "Training Time": training_time,
            }
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Training Time: "
            f"{training_time:.2f} seconds"
        )

    # ============================================================
    # 6. Display Results
    # ============================================================

    print("\n" + "=" * 60)
    print("HOG + Linear SVM Hyperparameter Results")
    print("=" * 60)

    print(
        f"{'C':>10} "
        f"{'Validation Accuracy':>25} "
        f"{'Training Time (s)':>20}"
    )

    print("-" * 60)

    for result in results:

        print(
            f"{result['C']:>10} "
            f"{result['Validation Accuracy']:>25.4f} "
            f"{result['Training Time']:>20.2f}"
        )

    # ============================================================
    # 7. Select Best C
    # ============================================================

    best_result = max(
        results,
        key=lambda result: result["Validation Accuracy"],
    )

    print("\n" + "=" * 60)
    print("Best Configuration")
    print("=" * 60)

    print(f"Best C: {best_result['C']}")
    print(
        f"Best Validation Accuracy: "
        f"{best_result['Validation Accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
