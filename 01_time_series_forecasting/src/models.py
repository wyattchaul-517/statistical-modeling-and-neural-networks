"""Model definitions for time series forecasting."""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train, y_train, n_estimators: int = 100, random_state: int = 42
) -> RandomForestRegressor:
    """Train a Random Forest Regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model