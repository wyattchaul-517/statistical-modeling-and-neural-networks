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

## Relevance to OIST Research

| Project Component | OIST Research Connection |
|:---|:---|
| Time series statistical modeling | MLDS Unit: high-dimensional statistical modeling, kernel methods |
| Matrix operations & eigenvalue decomposition | MLDS Unit: geometric machine learning, optimal transport |
| CNN implementation in PyTorch | MLDS Unit: deep learning models, GNN |
| Feature engineering pipeline | Biological Nonlinear Dynamics Data Science Unit (Prof. Gerald Pao): time-series analysis |

## Setup

```bash
pip install -r requirements.txt