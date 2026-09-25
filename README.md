# Clothing Image Classification Using HOG, Linear SVM, and CNN

Pattern Recognition course project using the Fashion-MNIST dataset.

## Project Goal

The goal of this project is to investigate and compare different
feature representations and classification approaches for clothing
image classification using the Fashion-MNIST dataset.

## Methods

The project evaluates three classification approaches:

1. Raw Pixel Features + Linear SVM
2. HOG Feature Extraction + Linear SVM
3. Convolutional Neural Network (CNN)

For the HOG + Linear SVM approach, the regularization parameter `C`
was tuned using the validation set.

## Dataset

Fashion-MNIST

- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 clothing classes
- 50,000 training images
- 10,000 validation images
- 10,000 test images

## Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- TP / TN / FP / FN per class
- Training Time
- Prediction Time

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scikit-image
- TensorFlow / Keras

## Project Structure

```text
pattern-recognition-fashion/
├── src/
│   ├── baseline.py
│   ├── hog_features.py
│   ├── hog_baseline.py
│   ├── tune_hog_svm.py
│   └── cnn.py
├── results/
├── models/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```
