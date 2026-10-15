import pandas as pd
import numpy as np
import pytest
from src.features import generate_features

def test_generate_features():
    # Create dummy DataFrame with sufficient periods for moving averages
    dates = pd.date_range("2023-01-01", periods=100)
    np.random.seed(42)
    df = pd.DataFrame({
        "Open": np.linspace(100, 200, 100),
        "High": np.linspace(105, 205, 100),
        "Low": np.linspace(95, 195, 100),
        "Close": np.linspace(100, 200, 100) + np.random.randn(100),
        "Volume": np.random.randint(1000, 5000, 100)
    }, index=dates)
    
    processed_df = generate_features(df)
    
    # Check if expected columns are generated
    expected_cols = [
        "EMA_20", "EMA_50", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", 
        "BB_Middle", "BB_Upper", "BB_Lower", "ATR_14", "Daily_Return", "Volatility_20"
    ]
    
    for col in expected_cols:
        assert col in processed_df.columns, f"Missing column: {col}"
        
    # Test length matches
    assert len(processed_df) == len(df), "DataFrame length should not change during basic feature generation"
