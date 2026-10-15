# Stock Market AI Platform

A production-ready, modular stock market AI platform using Python, PyTorch, and XGBoost. The platform downloads historical stock data from Yahoo Finance, performs feature engineering using technical indicators (RSI, MACD, EMA), trains models to predict next-day stock price direction, and evaluates performance.

---

## 1. Project Directory Structure

The project is structured modularly to follow production-grade software engineering best practices:

```
D:/Projects/Stock Market Projects/
├── models/                      # Folder to store serialized models (.json, .pt, scaler, and comparisons)
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Handles downloading data from Yahoo Finance via yfinance
│   ├── features.py              # Computes technical indicators: RSI, MACD, EMA, returns
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py     # XGBoost classifier model trainer and predictor
│   │   └── lstm_model.py        # PyTorch LSTM neural network model for sequence prediction
│   ├── evaluation.py            # Calculates classification metrics (Accuracy, Precision, F1, ROC-AUC)
│   └── pipeline.py              # Coordinates pipeline orchestration (data -> features -> train -> evaluate)
├── main.py                      # CLI entry point to run training and evaluation
├── requirements.txt             # Python project dependencies
└── README.md                    # Project documentation (this file)
```

---

## 2. Technical Features

The following technical indicators and feature transformations are engineered in `src/features.py`:

- **Exponential Moving Averages (EMA)**: Generates 9-day, 21-day, and 50-day EMA trends.
- **Relative Strength Index (RSI)**: Computes standard 14-day momentum.
- **Moving Average Convergence Divergence (MACD)**: Computes the MACD line (12-day vs. 26-day EMA), MACD signal (9-day EMA), and MACD histogram.
- **Volume Ratio**: The ratio of the current day's volume to the 10-day moving average volume.
- **Log Returns**: Computes daily log return features.
- **Price-to-EMA ratios**: Measure distance between the close price and the EMA lines.

### Target Definition
The target variable is **Stock Direction**:
- `1` (Up): If the next close price is higher than the current close price.
- `0` (Down): If the next close price is lower than or equal to the current close price.

---

## 3. Models

### A. XGBoost Classifier
- Standard gradient boosted trees using the `xgboost` framework.
- Optimized hyperparameters to prevent overfitting on financial market noise (e.g. learning rate of 0.05, max depth of 4, subsampling, and early stopping on validation loss).
- Serialized to `models/xgboost_model.json`.

### B. PyTorch LSTM (Long Short-Term Memory)
- Designed for time-series sequences. Takes a rolling window of historical bars (default: 10 days) to predict the next direction.
- Preprocesses and scales features using a `StandardScaler` fitted *only* on the training dataset to prevent lookahead/data leakage.
- Features early stopping based on validation loss.
- Weights saved as `models/lstm_model.pt`, scaler saved as `models/lstm_model_scaler.pkl`, and configuration saved as `models/lstm_model_config.json`.

---

## 4. Installation

Ensure you have Python Launcher (`py`) or Python 3.10+ installed.

Run the package installation using the system Python interpreter:
```bash
py -m pip install -r requirements.txt
```

---

## 5. Usage

To run the full end-to-end training and evaluation pipeline, use the `main.py` entrypoint script.

### Basic Run (Defaults to Apple - AAPL, 2020 to 2026):
```bash
py main.py
```

### Custom Run Options:
You can customize the ticker, dates, validation splits, sequence length, and storage location:
```bash
py main.py --ticker MSFT --start-date 2018-01-01 --end-date 2025-12-31 --seq-length 15 --models-dir ./my_models
```

### Command Line Arguments:
* `--ticker`: Stock ticker symbol (default: `AAPL`).
* `--start-date`: Start date for historical data download in `YYYY-MM-DD` (default: `2020-01-01`).
* `--end-date`: End date for historical data download in `YYYY-MM-DD` (default: `2026-01-01`).
* `--train-ratio`: Ratio of data allocated to training (default: `0.7`).
* `--val-ratio`: Ratio of data allocated to validation (default: `0.15`). The remainder is allocated to testing.
* `--seq-length`: Number of lookback days for PyTorch LSTM sequence (default: `10`).
* `--models-dir`: Output folder path where model checkpoints and scalers are saved (default: `./models`).

---

## 6. Evaluation Metrics

Both models are evaluated on the test set. To ensure a fair comparison:
1. Data splitting is chronological (no future data leakage in train/validation).
2. Evaluation on the test set is sliced to exclude the first `seq_length - 1` days (since LSTM needs a lookback sequence and cannot predict the very beginning of the test set).
3. Evaluated metrics include:
   - **Accuracy**: Proportion of correctly predicted directions.
   - **Precision**: Accuracy of the positive ("Up") predictions.
   - **Recall**: Proportion of true positive movements detected.
   - **F1-Score**: Harmonic mean of Precision and Recall.
   - **ROC-AUC**: Model's ability to distinguish between "Up" and "Down" directions.
   - **Confusion Matrix**: Logs True Positives, True Negatives, False Positives, and False Negatives.
