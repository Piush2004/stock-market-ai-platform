import os
import glob
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_ema(df: pd.DataFrame, column: str = "Close", spans: list = [20, 50]) -> pd.DataFrame:
    """Calculates Exponential Moving Average (EMA)."""
    df = df.copy()
    for span in spans:
        df[f"EMA_{span}"] = df[column].ewm(span=span, adjust=False).mean()
    return df

def calculate_rsi(df: pd.DataFrame, column: str = "Close", period: int = 14) -> pd.DataFrame:
    """Calculates the Relative Strength Index (RSI)."""
    df = df.copy()
    delta = df[column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    """Calculates MACD."""
    df = df.copy()
    ema_fast = df[column].ewm(span=12, adjust=False).mean()
    ema_slow = df[column].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df

def calculate_bollinger_bands(df: pd.DataFrame, column: str = "Close", period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Calculates Bollinger Bands."""
    df = df.copy()
    sma = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()
    df["BB_Middle"] = sma
    df["BB_Upper"] = sma + (std * std_dev)
    df["BB_Lower"] = sma - (std * std_dev)
    return df

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculates Average True Range (ATR)."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    df[f"ATR_{period}"] = true_range.rolling(window=period).mean()
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all requested technical features.
    """
    data = df.copy()
    
    # Existing / Requested Moving Averages
    data = calculate_ema(data, column="Close", spans=[20, 50])
    
    # RSI
    data = calculate_rsi(data, column="Close", period=14)
    
    # MACD
    data = calculate_macd(data, column="Close")
    
    # Bollinger Bands
    data = calculate_bollinger_bands(data, column="Close", period=20)
    
    # ATR
    data = calculate_atr(data, period=14)
    
    # Daily Returns
    data["Daily_Return"] = data["Close"].pct_change()
    
    # Volatility (20-day rolling standard deviation of daily returns)
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20).std()
    
    return data

def process_all_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
    """
    Iterates over all CSV files in raw_dir, calculates features, 
    and saves them to processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    if not files:
        logger.warning(f"No CSV files found in {raw_dir}")
        return

    logger.info(f"Found {len(files)} files to process.")
    
    for file_path in files:
        ticker = os.path.basename(file_path).replace(".csv", "")
        logger.info(f"Processing features for {ticker}...")
        
        try:
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            if df.empty:
                logger.warning(f"File {file_path} is empty. Skipping.")
                continue
                
            processed_df = generate_features(df)
            
            # Optional: drop NaNs created by rolling windows to have clean data
            # processed_df = processed_df.dropna()
            
            out_path = os.path.join(processed_dir, f"{ticker}_features.csv")
            processed_df.to_csv(out_path, index=True)
            logger.info(f"Saved processed data to {out_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    process_all_data()
