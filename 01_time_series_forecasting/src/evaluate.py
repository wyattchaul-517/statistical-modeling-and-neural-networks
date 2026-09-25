"""Evaluation utilities for forecasting models."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    """Compute MAE and RMSE for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} — MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return {'mae': mae, 'rmse': rmse}


def compare_models(results: dict) -> None:
    """Print a formatted comparison of model results."""
    print("\n" + "=" * 40)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8}")
    print("=" * 40)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['mae']:>8.2f} {metrics['rmse']:>8.2f}")
    print("=" * 40)