import os
import logging
import pandas as pd
from typing import Dict, Any, Tuple

from src.data_loader import download_stock_data
from src.features import generate_features
from src.models.xgboost_model import XGBoostModelWrapper
from src.models.lstm_model import LSTMModelWrapper
from src.evaluation import compute_regression_metrics, print_regression_summary, generate_comparison_df

logger = logging.getLogger(__name__)

def run_training_pipeline(
    ticker: str,
    start_date: str,
    end_date: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seq_length: int = 60,
    models_dir: str = "./models"
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Orchestrates the price forecasting pipeline:
    1. Downloads stock data
    2. Engineers technical features and regression targets
    3. Splits data chronologically
    4. Trains XGBoost Regressor and PyTorch LSTM Regressor
    5. Saves trained models and plots LSTM training loss curve
    6. Evaluates and compares performance on test set
    """
    logger.info("Initializing Stock Market Price Forecasting Pipeline...")
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Download stock data
    df_raw = download_stock_data(ticker, start_date, end_date)
    
    # 2. Feature engineering
    df_features, feature_cols = generate_features(df_raw, forecast_horizon=1)
    
    # 3. Chronological splitting (Time-series split)
    n = len(df_features)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_features.iloc[:train_end]
    val_df = df_features.iloc[train_end:val_end]
    test_df = df_features.iloc[val_end:]
    
    logger.info(
        f"Data split chronologically:\n"
        f"  Total records: {n}\n"
        f"  Train set:     {len(train_df)} ({train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')})\n"
        f"  Val set:       {len(val_df)} ({val_df.index[0].strftime('%Y-%m-%d')} to {val_df.index[-1].strftime('%Y-%m-%d')})\n"
        f"  Test set:      {len(test_df)} ({test_df.index[0].strftime('%Y-%m-%d')} to {test_df.index[-1].strftime('%Y-%m-%d')})"
    )
    
    X_train, y_train = train_df[feature_cols], train_df["Target"]
    X_val, y_val = val_df[feature_cols], val_df["Target"]
    X_test, y_test = test_df[feature_cols], test_df["Target"]
    
    # 4. Train XGBoost Regressor
    xgb_wrapper = XGBoostModelWrapper()
    xgb_wrapper.train(X_train, y_train, eval_set=(X_val, y_val))
    
    # Save XGBoost
    xgb_path = os.path.join(models_dir, "xgboost_model.json")
    xgb_wrapper.save(xgb_path)
    
    # 5. Train LSTM Regressor (with sequence length 60, early stopping, GPU if available, and saving best model)
    lstm_wrapper = LSTMModelWrapper(input_dim=len(feature_cols), config={"seq_length": seq_length})
    
    loss_plot_path = os.path.join(models_dir, "lstm_loss_plot.png")
    lstm_wrapper.train(X_train, y_train, X_val, y_val, loss_plot_path=loss_plot_path)
    
    # Save LSTM models and scalers
    lstm_wrapper.save(models_dir, model_name="lstm_model")
    
    # 6. Evaluation and Inference
    logger.info("Running predictions on test set...")
    
    # XGBoost predictions
    xgb_preds = xgb_wrapper.predict(X_test)
    
    # LSTM predictions (returns actual price predictions, padded with nan for first seq_length-1 entries)
    lstm_preds = lstm_wrapper.predict(X_test)
    
    # Slice evaluation window to exclude first seq_length - 1 lookback steps
    eval_start_idx = seq_length - 1
    y_test_eval = y_test.iloc[eval_start_idx:].values
    current_close_eval = test_df["Close"].iloc[eval_start_idx:].values
    
    xgb_preds_eval = xgb_preds[eval_start_idx:]
    lstm_preds_eval = lstm_preds[eval_start_idx:]
    
    # Compute Metrics
    xgb_metrics = compute_regression_metrics(y_test_eval, xgb_preds_eval, current_close_eval)
    lstm_metrics = compute_regression_metrics(y_test_eval, lstm_preds_eval, current_close_eval)
    
    # Log summaries
    print_regression_summary("XGBoost Regressor", xgb_metrics)
    print_regression_summary("PyTorch LSTM Regressor", lstm_metrics)
    
    # Compare
    comparison_df = generate_comparison_df(xgb_metrics, lstm_metrics)
    
    # Log feature importances
    importances = xgb_wrapper.get_feature_importances(feature_cols)
    logger.info(f"XGBoost Feature Importances:\n{importances.to_string()}")
    
    results = {
        "xgboost_metrics": xgb_metrics,
        "lstm_metrics": lstm_metrics,
        "xgb_feature_importances": importances
    }
    
    return results, comparison_df
