import os
import glob
import pandas as pd
import numpy as np
import logging
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

class XGBoostDirectionModel:
    def __init__(self, models_dir: str = "models", data_dir: str = "data/processed"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.features = [
            "EMA_20", "EMA_50", "RSI_14", "MACD", "MACD_Signal", "MACD_Hist", 
            "BB_Middle", "BB_Upper", "BB_Lower", "ATR_14", "Daily_Return", "Volatility_20"
        ]

    def prepare_data(self) -> pd.DataFrame:
        """
        Loads all processed CSV files, creates the target variable, 
        and concatenates them into a single dataset.
        """
        files = glob.glob(os.path.join(self.data_dir, "*_features.csv"))
        if not files:
            logger.error(f"No processed data found in {self.data_dir}.")
            return pd.DataFrame()

        all_data = []
        for file_path in files:
            try:
                df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
                
                # Ensure the data is sorted chronologically
                df = df.sort_index()
                
                # Target: 1 if tomorrow's Close > today's Close, else 0
                df["Next_Close"] = df["Close"].shift(-1)
                df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
                
                # Dynamically add RAG features if they exist
                rag_cols = [col for col in df.columns if col.startswith("RAG_")]
                for col in rag_cols:
                    if col not in self.features:
                        self.features.append(col)
                        
                # Drop rows with NaN (due to rolling windows or the shift at the end)
                # Keep only features and target
                cols_to_keep = self.features + ["Target"]
                df = df.dropna(subset=cols_to_keep)
                
                all_data.append(df[cols_to_keep])
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                
        if not all_data:
            return pd.DataFrame()
            
        combined_df = pd.concat(all_data)
        # Sort by date since we concatenated multiple tickers
        combined_df = combined_df.sort_index()
        return combined_df

    def train_and_evaluate(self):
        """
        Trains the XGBoost model and evaluates it using chronological split.
        """
        logger.info("Preparing data for XGBoost...")
        df = self.prepare_data()
        
        if df.empty:
            logger.error("Data preparation failed. Cannot train model.")
            return
            
        X = df[self.features]
        y = df["Target"]
        
        logger.info(f"Total dataset size: {len(df)} samples.")
        
        # Chronological train-test split (80-20)
        split_idx = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info("=== XGBoost Direction Model Evaluation ===")
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info("==========================================")
        
        # Save model
        model_path = os.path.join(self.models_dir, "xgboost_direction_model.json")
        self.model.save_model(model_path)
        logger.info(f"Model saved successfully to {model_path}")
        
        # Feature Importance
        importances = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        logger.info(f"Feature Importances:\n{importances.to_string(index=False)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    pipeline = XGBoostDirectionModel()
    pipeline.train_and_evaluate()
