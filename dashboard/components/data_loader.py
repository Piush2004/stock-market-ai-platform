import os
import streamlit as st
import pandas as pd

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]
TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "NVDA": "NVIDIA Corp.",
    "TSLA": "Tesla Inc.",
    "MSFT": "Microsoft Corp.",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
}
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"

@st.cache_data(ttl=300)
def load_raw_data(ticker: str) -> pd.DataFrame:
    """Loads raw OHLCV data for a ticker."""
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    return pd.DataFrame()

@st.cache_data(ttl=300)
def load_processed_data(ticker: str) -> pd.DataFrame:
    """Loads processed data with technical indicators."""
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col="Date", parse_dates=True)
    return pd.DataFrame()

def get_system_status() -> dict:
    """Returns status of all subsystems."""
    status = {}
    status["raw_data"] = all(os.path.exists(os.path.join(RAW_DIR, f"{t}.csv")) for t in TICKERS)
    status["processed_data"] = all(os.path.exists(os.path.join(PROCESSED_DIR, f"{t}_features.csv")) for t in TICKERS)
    status["xgboost_model"] = os.path.exists(os.path.join(MODELS_DIR, "xgboost_direction_model.json"))
    status["chroma_db"] = os.path.exists("data/chroma_db")
    return status
