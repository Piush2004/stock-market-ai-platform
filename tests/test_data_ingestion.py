import pandas as pd
import pytest
import os
from src.data_ingestion import DataIngestionPipeline

def test_data_validation():
    # Setup
    test_dir = "tests/mock_raw"
    pipeline = DataIngestionPipeline(raw_data_dir=test_dir)
    
    # Test 1: Empty dataframe should fail
    assert not pipeline.validate_data(pd.DataFrame(), "MOCK_EMPTY")
    
    # Test 2: Valid dataframe should pass
    valid_df = pd.DataFrame({
        "Open": [100.0, 101.0], 
        "High": [105.0, 106.0], 
        "Low": [95.0, 96.0], 
        "Close": [102.0, 103.0], 
        "Volume": [1000, 1500]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
    
    assert pipeline.validate_data(valid_df, "MOCK_VALID")
    
    # Test 3: Missing columns should fail
    invalid_df = pd.DataFrame({"Open": [100.0], "Close": [102.0]})
    assert not pipeline.validate_data(invalid_df, "MOCK_MISSING_COLS")
    
    # Test 4: Negative prices should fail
    neg_df = valid_df.copy()
    neg_df.loc["2023-01-01", "Close"] = -5.0
    assert not pipeline.validate_data(neg_df, "MOCK_NEG_PRICE")
    
    # Test 5: Unsorted index should fail
    unsorted_df = pd.DataFrame({
        "Open": [100.0, 101.0], 
        "High": [105.0, 106.0], 
        "Low": [95.0, 96.0], 
        "Close": [102.0, 103.0], 
        "Volume": [1000, 1500]
    }, index=pd.to_datetime(["2023-01-02", "2023-01-01"]))
    
    assert not pipeline.validate_data(unsorted_df, "MOCK_UNSORTED")
