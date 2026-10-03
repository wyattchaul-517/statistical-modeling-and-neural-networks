# Statistical Modeling and Neural Networks

A computational study connecting classical statistical modeling with modern neural network architectures.

## Overview

This project explores two complementary themes in machine learning:

1. **Statistical modeling for time series**  
   Feature engineering, model comparison, and the mathematical foundations of linear regression.

2. **Neural network architectures for image classification**  
   Implementing and analyzing a convolutional neural network for handwritten digit recognition.

The project emphasizes understanding methods from first principles rather than treating machine learning models as black boxes. This includes closed-form ordinary least squares, eigendecomposition, and explicit implementation of a CNN training pipeline.

The project also explores how these empirical results connect to broader questions about model complexity, generalization, and information-theoretic analysis.

---

## Modules

### 1. Time Series Forecasting with Statistical Learning

The first module uses the **AirPassengers** dataset to study time-series forecasting as a supervised learning problem.

#### Methods

- Lag feature engineering
- Linear Regression
- Random Forest Regression
- Closed-form Ordinary Least Squares (OLS)
- Covariance matrix eigendecomposition
- Principal Component Analysis (PCA)
- Model evaluation using MAE and RMSE

#### Results

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 17.19 | 20.76 |
| Random Forest | 28.99 | 38.17 |

Linear Regression performs better on this forecasting task. The difference is consistent with the fact that tree-based models have limited ability to extrapolate beyond the range represented in their training data.

---

### 2. Handwritten Digit Recognition with PyTorch

The second module implements a convolutional neural network for handwritten digit classification using **MNIST**.

#### Architecture

- 2 convolutional layers
- 2 fully connected layers
- Dropout
- ReLU activations
- Cross-entropy loss
- Mini-batch gradient descent

The model was trained for 5 epochs on 60,000 MNIST training samples.

#### Results

**Test accuracy: 99.17%**

- Test samples: 10,000
- Misclassified samples: 83
- Per-class accuracy: approximately 98.7%–99.6%

The error analysis shows that the most common mistakes occur between visually similar digits, including:

- 7 → 2
- 9 → 7
- 3 → 2
- 4 → 9
- 8 → 2

---

## Mathematical Foundations

A central goal of this project is to make the mathematical structure behind the models explicit.

### Closed-form Ordinary Least Squares

The linear regression model is also implemented using the normal equation:

$$
\hat{\beta} = (X^T X)^{-1}X^T y
$$

The NumPy implementation is compared against `scikit-learn`.

The maximum absolute difference between the two implementations is:

$$
4.88 \times 10^{-12}
$$

This provides a numerical verification of the closed-form implementation.

---

### Covariance Matrix and Eigendecomposition

The covariance structure of the lagged time-series features is analyzed through eigendecomposition:

$$
\Sigma = Q \Lambda Q^T
$$

The analysis verifies the expected symmetry and positive semi-definite structure of the covariance matrix.

The first three principal components explain approximately **96% of the total variance**, with the first principal component alone explaining approximately **85.4%**.

This indicates that the lagged representation contains substantial low-dimensional structure.

---

## Information-Theoretic Perspective

The project also explores an information-theoretic perspective on machine learning generalization.

The current implementation does **not** attempt to establish a formal information-theoretic generalization bound. Instead, the repository provides a foundation for future experiments involving quantities such as:

- Mutual information
- Sibson α-mutual information
- Maximal leakage
- Generalization gaps
- Model complexity and regularization

These quantities are particularly interesting because they provide alternative ways of studying the relationship between the information contained in a learned model and its ability to generalize.

The current dataset and experimental design impose important limitations, discussed below.

---

## Repository Structure

```text
statistical-modeling-and-neural-networks/
│
├── 01_time_series_forecasting/
│   ├── notebooks/
│   │   └── EDA, modeling, and mathematical analysis
│   │
│   ├── src/
│   │   └── Data loading, feature engineering, and models
│   │
│   └── results/
│       └── Figures and metric JSON files
│
├── 02_mnist_pytorch/
│   ├── notebooks/
│   │   └── Training and error analysis
│   │
│   ├── src/
│   │   └── CNN model, training loop, and evaluation
│   │
│   └── results/
│       └── Figures, model weights, and metrics
│
├── requirements.txt
└── README.md

## Limitations and Future Work

### Sample Size

The time-series analysis uses 120 training observations.

While this is sufficient for the current model comparison, it is relatively small for reliable estimation of information-theoretic quantities, particularly when continuous variables are discretized into bins.

A natural extension would be to use larger datasets and investigate the stability of information-theoretic estimates under different discretization strategies.

---

### Non-Stationarity

The AirPassengers series contains strong trend and annual seasonality.

Many classical information-theoretic generalization results are formulated under independent and identically distributed (i.i.d.) assumptions. The temporal dependence and non-stationarity of this dataset therefore make a direct application of such bounds inappropriate without additional assumptions.

Future work could investigate information-theoretic generalization in non-i.i.d. settings, including settings with temporal dependence or Markov structure.

---

### Model Scope

The current project focuses on:

- Linear Regression
- Random Forest Regression
- A small convolutional neural network

Future extensions could investigate additional model classes, such as:

- Ridge Regression
- Kernel methods
- Gaussian Processes
- Deeper neural networks
- Graph Neural Networks

A larger experimental setup would also make it possible to study how model complexity, regularization, and information-theoretic quantities interact across different architectures.

---

## Author

**Lei Zhou**  
BSc Software Engineering  
University of Gothenburg