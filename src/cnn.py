from sklearn.datasets import fetch_openml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from sklearn.metrics import classification_report, confusion_matrix


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

    data = fetch_openml(
    name="Fashion-MNIST",
    version=1,
    as_frame=False,
    )

    X = data.data
    y = np.asarray(data.target, dtype=np.int64)

    X = X.astype(np.float32) / 255.0

    # Original Fashion-MNIST:
    print("Label dtype:", y.dtype)
    print("Label sample:", y[:10])
    # 60,000 training samples + 10,000 test samples
    X_train = X[:50000]
    y_train = y[:50000]

    X_val = X[50000:60000]
    y_val = y[50000:60000]

    X_test = X[60000:]
    y_test = y[60000:]

    # CNN expects image shape:
    # (samples, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    print("Training samples:", X_train.shape)
    print("Validation samples:", X_val.shape)
    print("Testing samples:", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test
def build_cnn():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

    model = build_cnn()

    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    print("\nTraining CNN...")

    training_start = time.time()

    history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1,
    )

    training_time = time.time() - training_start

    print(f"\nTraining time: {training_time:.2f} seconds")
    print("\nEvaluating on test set...")

    prediction_start = time.time()

    y_prob = model.predict(
        X_test,
        batch_size=128,
        verbose=0,
    )

    prediction_time = time.time() - prediction_start

    y_pred = np.argmax(y_prob, axis=1)

    test_accuracy = np.mean(y_pred == y_test)

    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=2,
        )
    )
    print("\nConfusion Matrix:")

    cm = confusion_matrix(y_test, y_pred)

    print(cm)
    print("\nPer-class TP / TN / FP / FN:")

    total = cm.sum()

    for i, class_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        print(
            f"{class_name:<15} "
            f"TP={tp:4d} "
            f"TN={tn:4d} "
            f"FP={fp:4d} "
            f"FN={fn:4d}"
        )
