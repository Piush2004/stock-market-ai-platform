import os
import logging
import yfinance as yf
import pandas as pd
from typing import List

logger = logging.getLogger(__name__)

class DataIngestionPipeline:
    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = raw_data_dir
        self.expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Ensure the raw data directory exists
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
    def get_file_path(self, ticker: str) -> str:
        """Returns the CSV file path for a given ticker."""
        return os.path.join(self.raw_data_dir, f"{ticker}.csv")

    def download_historical_data(self, ticker: str, period: str = "max") -> pd.DataFrame:
        """
        Downloads historical OHLCV data for a ticker using yfinance.
        """
        logger.info(f"Downloading historical data for {ticker} (Period: {period})...")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(period=period)
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}. Check if the ticker symbol is correct.")
            return df
            
        # Keep only OHLCV
        df = df[self.expected_columns]
        df.index = pd.to_datetime(df.index).tz_localize(None) # Remove timezone for consistency
        return df

    def validate_data(self, df: pd.DataFrame, ticker: str) -> bool:
        """
        Performs data validation checks.
        """
        is_valid = True
        
        # Check if empty
        if df.empty:
            logger.error(f"Validation failed for {ticker}: DataFrame is empty.")
            return False

        # Check for expected columns
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Validation failed for {ticker}: Missing columns: {missing_cols}")
            is_valid = False

        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Validation warning for {ticker}: Missing values found.\n{null_counts[null_counts > 0]}")
            # We don't fail validation strictly on nulls, but we log it. It could be handled via ffill downstream.

        # Check for negative prices
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns and (df[col] < 0).any():
                logger.error(f"Validation failed for {ticker}: Negative prices found in {col}.")
                is_valid = False
                
        # Check if index is sorted
        if not df.index.is_monotonic_increasing:
            logger.error(f"Validation failed for {ticker}: Index is not strictly monotonically increasing.")
            is_valid = False

        if is_valid:
            logger.info(f"Data validation passed for {ticker}.")
            
        return is_valid

    def update_data(self, ticker: str):
        """
        Automated update function. 
        If the data file exists, it finds the last date and downloads only new data.
        If it doesn't exist, it downloads historical data.
        """
        file_path = self.get_file_path(ticker)
        
        if not os.path.exists(file_path):
            logger.info(f"No existing data found for {ticker}. Running initial full download.")
            df = self.download_historical_data(ticker, period="5y") # Adjust max/5y as needed
            
            if not df.empty and self.validate_data(df, ticker):
                df.to_csv(file_path, index=True, index_label="Date")
                logger.info(f"Saved initial data for {ticker} to {file_path}")
            return

        # If data exists, update it
        try:
            existing_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        except Exception as e:
            logger.error(f"Failed to read existing data for {ticker}: {e}")
            return
            
        if existing_df.empty:
            logger.warning(f"Existing file for {ticker} is empty. Redownloading...")
            df = self.download_historical_data(ticker, period="5y")
            if not df.empty and self.validate_data(df, ticker):
                df.to_csv(file_path, index=True, index_label="Date")
            return

        last_date = existing_df.index[-1]
        logger.info(f"Existing data for {ticker} found up to {last_date.date()}. Fetching new data...")
        
        # We fetch last 1 month to ensure we get any overlapping corrected recent data, or use 'start'
        start_date = last_date.strftime('%Y-%m-%d')
        ticker_obj = yf.Ticker(ticker)
        new_df = ticker_obj.history(start=start_date)
        
        if new_df.empty:
            logger.info(f"No new data available for {ticker} since {start_date}.")
            return
            
        new_df = new_df[self.expected_columns]
        new_df.index = pd.to_datetime(new_df.index).tz_localize(None)
        
        # Combine and drop duplicates
        combined_df = pd.concat([existing_df, new_df])
        # Keep the last entry for any duplicate dates to get the most updated prices
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df = combined_df.sort_index()
        
        if self.validate_data(combined_df, ticker):
            combined_df.to_csv(file_path, index=True, index_label="Date")
            logger.info(f"Successfully updated data for {ticker}. Total records: {len(combined_df)}.")

    def run_pipeline(self, tickers: List[str]):
        """
        Runs the full ingestion and update pipeline for a list of tickers.
        """
        logger.info(f"Starting data ingestion pipeline for {len(tickers)} tickers...")
        for ticker in tickers:
            logger.info("-" * 40)
            try:
                self.update_data(ticker)
            except Exception as e:
                logger.error(f"Pipeline failed for {ticker}: {str(e)}")
        logger.info("-" * 40)
        logger.info("Data ingestion pipeline completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    # Target tickers
    TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "RELIANCE.NS", "TCS.NS"]
    
    pipeline = DataIngestionPipeline(raw_data_dir="data/raw")
    pipeline.run_pipeline(TICKERS)
