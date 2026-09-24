"""Data loading and preprocessing for time series forecasting."""

import pandas as pd
import seaborn as sns


def load_air_passengers() -> pd.DataFrame:
    """
    Load the Air Passengers dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and a single 'value' column
        representing monthly passenger counts (1949-1960).
    """
    df = sns.load_dataset('flights')
    df['date'] = pd.to_datetime(
    df['year'].astype(str) + '-' + df['month'].astype(str) + '-01',
    format='%Y-%b-%d'
    )
    df = df.set_index('date').rename(columns={'passengers': 'value'})
    return df[['value']].sort_index()


def train_test_split_time_series(
    df: pd.DataFrame, test_size: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split for time series data.
    """
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


if __name__ == "__main__":
    data = load_air_passengers()
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(data.head())
    print(data.tail())