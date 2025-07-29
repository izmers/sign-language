# Sign Language Image Classifier

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Framework](https://img.shields.io/badge/framework-PyTorch%20|%20ScikitLearn-blue)]()
[![Dataset](https://img.shields.io/badge/dataset-9,680%20images-lightgrey)]()

A project to classify 9,680 grayscale sign language images (letters a–z and digits 0–9) into 36 classes using two different approaches:

1. **Convolutional Neural Network (CNN)** implemented with PyTorch.
2. **Support Vector Machine (SVM)** with PCA for dimensionality reduction and grid search for hyperparameter tuning.

---

## Overview

This repository contains two implementations for classifying sign language gestures from grayscale images into 36 classes (digits `0–9` and letters `a–z`). The first approach leverages a deep learning model (CNN) in PyTorch, and the second uses a classical machine learning pipeline (SVM) with principal component analysis (PCA).

## Dataset

- **Images**: 9,680 grayscale PNG images of size 64×64.
- **Labels**: A CSV file (`data/labels.csv`) mapping each image filename to its class label (digit or letter).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sign-language-classifier.git
   cd sign-language-classifier
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> **Dependencies** include PyTorch, torchvision, scikit-learn, scikit-image, pandas, numpy, and OpenCV.

## Usage

### 1. Train the CNN model

```bash
python src/cnn_training.py \
  --data-path data/images \
  --labels-file data/labels.csv \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.0001
```

This will save a trained model to `cnn_model.pt`.

### 2. Train the SVM model

```bash
python src/svm_training.py \
  --data-path data/images \
  --labels-file data/labels.csv \
  --test-size 0.2
```

This will output the best SVM parameters via grid search and save the classifier to `svm_model.p`.

### Method 1: Convolutional Neural Network

- **Architecture**: Two convolutional layers with ReLU and max pooling, followed by dropout and a fully connected layer for 36 outputs.
- **Custom Dataset**: `SignLangDataset` reads images and labels from CSV, returning PyTorch tensors.
- **Training**:
  - Learning rate: 0.0001
  - Epochs: 10
  - Batch size: 32
  - Optimizer: Adam with weight decay
  - Loss: CrossEntropyLoss
- **Model Compression**: Initial model was ~130 MB. After pruning layers and reducing channels, final size is ~4 MB.

### Method 2: Support Vector Machine

- **Preprocessing**:
  - Resize images to 64×64 and flatten to 1D arrays.
  - Labels converted to integers (`0–9` for digits, `10–35` for `a–z`).
- **Dimensionality Reduction**: PCA retaining 95% variance.
- **Model Training**:
  - Split: 80% train, 20% test (stratified).
  - Hyperparameters: `C` ∈ {1, 10}, `gamma` ∈ {0.01, 0.001}.
  - Grid search with 5‑fold cross‑validation.
- **Evaluation**: Accuracy reported on held‑out set.

---
