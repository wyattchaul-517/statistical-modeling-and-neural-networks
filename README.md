# Statistical Modeling and Neural Networks

A computational study connecting classical statistical modeling with modern neural network architectures.

## Overview

This project explores two themes in machine learning:

1. **Statistical modeling for time series**  
   Feature engineering, model comparison, and the mathematical foundations of linear regression.

2. **Neural network architectures for image classification**  
   A convolutional neural network trained on handwritten digits.

The goal is to implement methods from first principles rather than treat them as black boxes. That includes the closed-form OLS solution, eigendecomposition, and a CNN training pipeline written from scratch.

The project also includes a small information-theoretic analysis of the time-series models. It is exploratory, and the scope limitation is discussed in that section.

## Modules

### 1. Time Series Forecasting

The first module uses the AirPassengers dataset to study forecasting as a supervised learning problem.

Methods:

- Lag feature engineering
- Linear Regression
- Random Forest Regression
- Closed-form Ordinary Least Squares
- Covariance matrix eigendecomposition
- Principal component analysis via eigendecomposition
- Evaluation using MAE and RMSE

Results:

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 17.19 | 20.76 |
| Random Forest | 28.99 | 38.17 |

Linear Regression does better on this task. A decision tree produces a piecewise-constant function, so it cannot return values outside the range of targets it saw during training. The test period extends beyond that range, and the Random Forest underpredicts. Linear Regression fits a linear combination of lag features and can extend past the training range.

### 2. Handwritten Digit Recognition

The second module implements a CNN for MNIST classification.

Architecture:

- 2 convolutional layers
- 2 fully connected layers
- Dropout
- ReLU activations
- Cross-entropy loss
- Mini-batch gradient descent

Trained for 5 epochs on 60,000 samples.

Results:

Test accuracy: 99.17%
Misclassified: 83 out of 10,000
Per-class accuracy: about 98.7% to 99.6%

The most common mistakes happen between visually similar digits:

- 7 → 2
- 9 → 7
- 3 → 2
- 4 → 9
- 8 → 2

## Mathematical Foundations

A central goal is to make the math behind the models explicit.

### Closed-form OLS

Linear regression is also implemented with the normal equation:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

The NumPy implementation is compared against scikit-learn. The maximum absolute difference is:

$$4.88 \times 10^{-12}$$

This verifies the manual implementation numerically.

### Covariance Matrix and Eigendecomposition

The covariance structure of the lagged features is analyzed through eigendecomposition:

$$\Sigma = Q \Lambda Q^T$$

The matrix is symmetric and positive semi-definite, as expected. The first three principal components cover about 96% of the total variance, with PC1 alone covering about 85.4%. The lagged representation has substantial low-dimensional structure.

## Information-Theoretic Exploratory Analysis

The project includes a small analysis using discrete Sibson α-mutual information between targets and model predictions, written as $I_\alpha(Y; \hat{Y})$.

A note on what this quantity is. In formal information-theoretic generalization bounds, the object of interest is $I(S; A(S))$, the mutual information between the training set and the learned model. That is not what gets computed here. What gets computed is how strongly the predictions depend on the target under a chosen discretization. The experiment is exploratory and does not try to validate any generalization bound.

The analysis covers four things:

- Baseline α-MI comparison between Linear Regression and Random Forest predictions
- Bin sensitivity with 4, 6, 8, and 10 quantile bins
- Ridge regularization with α from 0.001 to 100 on standardized features
- Random Forest depth from 2 up to unrestricted

Observations:

At every bin count, RF predictions carry more α-MI about the target than LR predictions. The ordering is stable, so comparing the two models at a fixed bin count is meaningful even though the absolute values depend on the bin count.

The Ridge sweep produces a U-shaped generalization gap with a minimum near α = 1.0, but α-MI barely moves along the sweep. Regularization strength inside a fixed model class does not seem to shift this diagnostic much.

For Random Forest depth, α-MI and the gap do not move together. From depth 2 to depth 5, α-MI climbs from 1.12 to 1.64 while the gap drops from 3756 to 1375. From depth 5 to depth 10, α-MI plateaus and the gap ticks up slightly.

The α-MI diagnostic separates model classes but does not track overfitting within a single class. Given the small dataset (120 training, 12 test) and the non-i.i.d. structure of the series, all estimates here are exploratory.

Results and tables are in `02_modeling.ipynb`.

## Repository Structure

```
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
```


## Limitations and Future Work

### Sample Size

The time-series analysis uses 120 training observations. This is enough for model comparison but small for reliable estimation of information-theoretic quantities, especially when continuous variables get discretized into bins. A larger dataset would allow a more stable estimate across different bin counts.

### Non-Stationarity

The AirPassengers series has a strong trend and yearly seasonality. Information-theoretic generalization bounds are usually stated for i.i.d. samples. The temporal dependence here means the bounds do not apply directly without extra assumptions. Extending the analysis to non-i.i.d. settings would be a natural next step.

### Model Scope

The current project covers Linear Regression, Random Forest Regression, and a small CNN. Adding kernel methods, Gaussian Processes, deeper networks, or graph neural networks would let us study how model complexity, regularization, and information-theoretic quantities interact across a wider range of models.

## Author

Lei Zhou  
BSc Software Engineering  
University of Gothenburg