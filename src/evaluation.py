import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_regression_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    current_close: np.ndarray
) -> Dict[str, float]:
    """
    Computes regression performance metrics and Directional Accuracy.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True next-day close prices.
    y_pred : np.ndarray
        Predicted next-day close prices.
    current_close : np.ndarray
        Close prices of the current day.
        
    Returns:
    --------
    Dict[str, float]
        Dictionary of computed evaluation metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    current_close = np.array(current_close)
    
    # Standard regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy: did the prediction guess the correct direction (up/down)?
    actual_diff = y_true - current_close
    predicted_diff = y_pred - current_close
    
    # 1 for up/no change, 0 for down
    actual_dir = (actual_diff > 0).astype(int)
    predicted_dir = (predicted_diff > 0).astype(int)
    
    directional_accuracy = np.mean(actual_dir == predicted_dir)
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "directional_accuracy": float(directional_accuracy)
    }

def print_regression_summary(model_name: str, metrics: Dict[str, Any]) -> None:
    """
    Logs and prints regression evaluation summary.
    """
    summary = (
        f"\n=========================================\n"
        f" EVALUATION SUMMARY: {model_name}\n"
        f"=========================================\n"
        f" Mean Squared Error (MSE):     {metrics['mse']:.4f}\n"
        f" Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n"
        f" Mean Absolute Error (MAE):     {metrics['mae']:.4f}\n"
        f" R-squared (R2 Score):          {metrics['r2']:.4f}\n"
        f" Directional Accuracy:          {metrics['directional_accuracy'] * 100:.2f}%\n"
        f"=========================================\n"
    )
    logger.info(summary)
    print(summary)

def generate_comparison_df(xgboost_metrics: Dict[str, Any], lstm_metrics: Dict[str, Any]) -> pd.DataFrame:
    """
    Creates a pandas DataFrame comparing regression metrics.
    """
    comparison_data = {
        "Metric": ["MSE", "RMSE", "MAE", "R2 Score", "Directional Accuracy"],
        "XGBoost": [
            xgboost_metrics["mse"],
            xgboost_metrics["rmse"],
            xgboost_metrics["mae"],
            xgboost_metrics["r2"],
            xgboost_metrics["directional_accuracy"]
        ],
        "LSTM": [
            lstm_metrics["mse"],
            lstm_metrics["rmse"],
            lstm_metrics["mae"],
            lstm_metrics["r2"],
            lstm_metrics["directional_accuracy"]
        ]
    }
    return pd.DataFrame(comparison_data)
