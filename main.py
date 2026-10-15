import os
import argparse
import logging
from datetime import datetime

from src.pipeline import run_training_pipeline

def setup_logging():
    """
    Sets up logging to both console and a log file.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file saved at: {log_file}")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Stock Market AI Platform Price Forecasting Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD) for training data")
    parser.add_argument("--end-date", type=str, default="2026-01-01", help="End date (YYYY-MM-DD) for training data")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of data for training (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Fraction of data for validation (default: 0.15)")
    parser.add_argument("--seq-length", type=int, default=60, help="Sequence length for LSTM model (default: 60)")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory to save models (default: ./models)")
    
    args = parser.parse_args()
    
    logging.info("Starting run with configuration:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")
        
    try:
        results, comparison_df = run_training_pipeline(
            ticker=args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seq_length=args.seq_length,
            models_dir=args.models_dir
        )
        
        logging.info("\nPipeline execution complete. Model Comparison Results:")
        logging.info(f"\n{comparison_df.to_string(index=False)}")
        
        # Save comparison results to models directory
        comparison_path = os.path.join(args.models_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        logging.info(f"Comparison metrics saved to {comparison_path}")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
