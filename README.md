# Statistical Modeling and Neural Networks

A computational study connecting classical statistical modeling with modern
neural network architectures, motivated by research directions in OIST's
Machine Learning and Data Science Unit (MLDS) and related computational units.

## Motivation

This project explores two complementary research themes:

1. **Statistical modeling for high-dimensional time series** — directly relevant to
   MLDS Unit research on kernel methods, statistical modeling, and optimal transport.
2. **Neural network architectures for structured data** — connecting to MLDS work on
   graph neural networks (GNN) and deep learning models.

## Modules

### 1. Time Series Forecasting with Statistical Learning
- Lag feature engineering and supervised learning formulation
- Linear Regression and Random Forest Regressor comparison
- **Mathematical foundations**: closed-form solution via matrix operations,
  eigenvalue decomposition of autocovariance matrices

### 2. Handwritten Digit Recognition with Neural Networks
- CNN implementation from scratch in PyTorch
- Complete training pipeline with validation monitoring
- Error analysis and misclassification visualization

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

Results are reproducible via `notebooks/03_math_foundations.ipynb`.

## Setup

```bash
pip install -r requirements.txt

Author
Lei Zhou — BSc Software Engineering, University of Gothenburg