## Exploring convutional layers through data and experiments

---

# Table of Contents
1. [Project Overview](#1Project-Overview)
2. [Dataset Description](#2Dataset-Description)
3. [Exploratory Data Analysis (EDA)](#3Exploratory-Data-Analysis-EDA)
4. [Baseline Model (Non-Convolutional)](#4Baseline-Model-Non-Convolutional)
5. [Convolutional Neural Network Architecture](#5Convolutional-Neural-Network-Architecture)

---

# 1. Project Overview

This project delves into the application of convolutional neural networks as architectural components rather than as a black box.
The primary objective of the tutorial is to examine the inductive biases of the convolutional layers, which are typically extended by learning parameters such as kernel size, depth, stride, and pool.

A convolutional model is created from scratch and compared to a non-convolutional baseline with a realistic image dataset. Controlled tests are carried out to evaluate the influence of individual parameters in the convolution process.

The current project is associated with an academic assignment, and the topic of interest is associated with architectural reasoning and hyperparameter tuning.

# 2. Dataset Description
- Dataset: (e.g. CIFAR-10 / Fashion-MNIST / custom dataset — specify here)
- Source: (TensorFlow Datasets / torchvision / Kaggle)
- Data Type: Image data (2D tensors with channels)
- Number of Classes: (e.g. 10)
- Image Shape: (e.g. 32×32×3)

Why this dataset?

- This dataset is appropriate for convolutional architectures because:
- It has strong spatial structure
- Local patterns (edges, textures, shapes) are meaningful
- Translation invariance is desirable
- Fully connected models struggle to scale efficiently

---

# 3. Exploratory Data Analysis (EDA)

The EDA focuses on understanding structure rather than exhaustive statistics:

- Dataset size and class distribution
- Image dimensions and number of channels
- Visualization of sample images per class

Preprocessing steps:

- Normalization
- Optional resizing
- Label encoding

See the notebook for visual examples and analysis.

---

# 4. Baseline Model (Non-Convolutional)

A simple neural network without convolutional layers is implemented as a reference point.

Architecture:

- Flatten layer
- Fully connected (Dense) layers
- ReLU activations
- Softmax output

Observations:

- High number of parameters
- Poor generalization compared to CNN
- No spatial inductive bias
- Sensitive to image dimensionality

This model establishes a baseline for comparison.

---

# 5. Convolutional Neural Network Architecture

The CNN is designed intentionally, not copied from a tutorial.

Design Choices:

- Convolutional layers: (e.g. 2–3 layers)
- Kernel size: (e.g. 3×3)
- Stride & Padding: (e.g. stride=1, same padding)
- Activation function: ReLU
- Pooling: (e.g. MaxPooling 2×2)
- Classifier: Flatten + Dense layers

Each choice is justified based on:

- Spatial locality
- Parameter efficiency
- Feature hierarchy

---

# 