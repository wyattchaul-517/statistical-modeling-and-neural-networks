"""Feature engineering for time series forecasting."""

import pandas as pd


def create_lag_features(
    df: pd.DataFrame, n_lags: int = 12, target_col: str = 'value'
) -> pd.DataFrame:
    """
    Create lag features to convert a time series into a supervised
    learning problem.
    """
    df_features = df.copy()
    for lag in range(1, n_lags + 1):
        df_features[f'lag_{lag}'] = df_features[target_col].shift(lag)
    df_features = df_features.dropna()
    return df_features


def get_feature_target_columns(
    df: pd.DataFrame, target_col: str = 'value'
) -> tuple[list[str], str]:
    """Return feature column names and target column name."""
    feature_cols = [c for c in df.columns if c.startswith('lag_')]
    return feature_cols, target_col