# Statistical Modeling and Neural Networks

A computational study connecting classical statistical modeling with modern
neural network architectures.

## Overview

This project explores two complementary themes in machine learning:

1. **Statistical modeling for time series** — feature engineering, model
   comparison, and the mathematical foundations behind linear regression.
2. **Neural network architectures for image classification** — implementing
   a CNN from scratch and analyzing its behavior on handwritten digits.
   
  The project emphasizes implementing methods from first principles (closed-form
  OLS, eigendecomposition, CNN from scratch) rather than treating them as black
  boxes. This mathematical foundation motivates an information-theoretic view
  of generalization (This is discussed in the Limitations section below).

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

## Limitations and Future Work

**Sample size.** The time-series analysis uses 120 training observations
(monthly data from 1950 to 1959). While sufficient for model comparison,
this sample size is too small for reliable estimation of information-theoretic
quantities such as Sibson α-mutual information or Maximal Leakage, which
require joint distributions over discretized spaces. A natural extension
with larger datasets would be to estimate such measures and compare them
to empirical generalization gaps.

**Non-stationarity.** The AirPassengers series exhibits a strong trend and
annual seasonality. The generalization bounds from information theory
typically assume i.i.d. samples, which does not hold here. Extending the
analysis to non-i.i.d. settings  would be a meaningful direction (for example via the concentration
inequalities developed for Markov chains).

**Model scope.** The current project focuses on two classical models
(linear regression and random forest) for the time-series task, and a
small CNN for image classification. Expanding to more expressive
architectures (kernel methods, Gaussian processes, or graph neural networks)
and analyzing their information-theoretic generalization behavior would
require substantially more data.


## Author

Lei Zhou — BSc Software Engineering, University of Gothenburg