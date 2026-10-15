import logging
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger(__name__)

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Downloads historical stock data from Yahoo Finance.

    Parameters:
    -----------
    ticker : str
        The ticker symbol of the stock (e.g., 'AAPL', 'MSFT').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    interval : str, default '1d'
        Data interval (e.g., '1d', '1wk', '1mo').

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the stock history with columns:
        Open, High, Low, Close, Volume, etc., with Date as the index.
    
    Raises:
    -------
    ValueError
        If no data is downloaded or columns are missing.
    """
    logger.info(f"Downloading historical data for {ticker} from {start_date} to {end_date} (interval: {interval})...")
    
    try:
        # Fetch the ticker data
        df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}' in the specified range.")
        
        # In newer versions of yfinance, columns can be multi-indexed if multiple tickers are queried,
        # but for single tickers it should be flat. We ensure columns are flat.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Downloaded data is missing required columns: {missing_cols}")
            
        # Clean any multi-index or name issues
        df.index.name = "Date"
        
        # Sort index just in case
        df = df.sort_index()
        
        logger.info(f"Successfully downloaded {len(df)} rows for {ticker}.")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading stock data for {ticker}: {str(e)}")
        raise e
