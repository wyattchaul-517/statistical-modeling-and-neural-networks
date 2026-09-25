# Statistical Modeling and Neural Networks

A computational study connecting classical statistical modeling with modern
neural network architectures.

## Overview

This project explores two complementary themes in machine learning:

1. **Statistical modeling for time series** — feature engineering, model
   comparison, and the mathematical foundations behind linear regression.
2. **Neural network architectures for image classification** — implementing
   a CNN from scratch and analyzing its behavior on handwritten digits.

## Modules

### 1. Time Series Forecasting with Statistical Learning

- Lag feature engineering to formulate forecasting as supervised learning
- Comparison of Linear Regression and Random Forest Regressor
- **Results**: Linear Regression achieves MAE 17.19 / RMSE 20.76, outperforming
  Random Forest (MAE 28.99 / RMSE 38.17) due to the latter's inability to
  extrapolate beyond the training range.

### 2. Handwritten Digit Recognition with Neural Networks

- SimpleCNN implemented from scratch in PyTorch (2 conv + 2 FC layers)
- Trained for 5 epochs on 60,000 MNIST samples
- **Results**: 99.17% test accuracy (83 errors out of 10,000)
- Confusion analysis reveals that misclassifications align with known
  handwritten-digit ambiguities (7↔2, 9↔7, 4↔9)

## Mathematical Foundations

The time series module demonstrates the underlying linear algebra explicitly:

- **Closed-form OLS**: The normal equation $\hat{\beta} = (X^T X)^{-1} X^T y$
  is implemented in NumPy and verified against scikit-learn
  (maximum absolute difference: $4.88 \times 10^{-12}$).
- **Autocovariance eigendecomposition**: The autocovariance matrix of the
  lagged series is decomposed as $\Sigma = Q \Lambda Q^T$, confirming
  symmetry and positive semi-definiteness.
- **Low-dimensional structure**: The top 3 principal components explain
  **96% of total variance** (PC1 alone: 85.4%), revealing that the series
  is dominated by a long-term trend plus annual seasonality.

## Repository Structure
   statistical-modeling-and-neural-networks/
   ├── 01_time_series_forecasting/
   │ ├── notebooks/ # EDA, modeling, mathematical foundations
   │ ├── src/ # Data loading, feature engineering, models
   │ └── results/ # Figures and metric JSONs
   └── 02_mnist_pytorch/
   ├── notebooks/ # Training and error analysis
   ├── src/ # CNN model, training loop, evaluation
   └── results/ # Figures, model weights, metrics

 ## Setup

```bash
pip install -r requirements.txt

Author
Lei Zhou — BSc Software Engineering, University of Gothenburg