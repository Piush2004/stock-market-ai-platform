import os
import json
import torch
import joblib
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    """
    Dataset class to convert tabular time-series features and regression targets into historical sequences.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 60):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        if len(self.X) < self.seq_length:
            return 0
        return len(self.X) - self.seq_length + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_length]
        y_val = self.y[idx + self.seq_length - 1]
        return x_seq, y_val


class StockLSTM(nn.Module):
    """
    LSTM architecture for stock price forecasting (regression).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape of x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Grab output of the final time step
        last_time_step = lstm_out[:, -1, :]
        out = self.dropout(last_time_step)
        predictions = self.fc(out)
        return predictions.squeeze(-1) # shape: (batch_size,)


class LSTMModelWrapper:
    def __init__(self, input_dim: int = 8, config: Dict[str, Any] = None):
        self.config = {
            "seq_length": 60,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 32,
            "weight_decay": 1e-4,
            "patience": 12,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        if config:
            self.config.update(config)
            
        self.device = torch.device(self.config["device"])
        self.model = StockLSTM(
            input_dim=input_dim,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)
        
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def fit_scalers(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits StandardScaler on features and targets, returning scaled arrays.
        """
        X_scaled = self.scaler.fit_transform(X_train)
        y_scaled = self.target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled
        
    def scale_features(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(X)

    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame, 
        y_val: pd.Series,
        loss_plot_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        Fits scaling, builds sequences, and trains the LSTM regression model.
        """
        logger.info(f"Preparing LSTM sequences... (seq_length={self.config['seq_length']})")
        
        # Fit/transform train, transform val
        X_train_scaled, y_train_scaled = self.fit_scalers(X_train, y_train)
        
        X_val_scaled = self.scale_features(X_val)
        y_val_scaled = self.target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, self.config["seq_length"])
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, self.config["seq_length"])
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Insufficient data to build sequences. Increase data length or decrease seq_length.")
            
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config["learning_rate"], 
            weight_decay=self.config["weight_decay"]
        )
        
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None
        
        logger.info(f"Training LSTM for stock forecasting on {self.device}...")
        
        for epoch in range(1, self.config["epochs"] + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    predictions = self.model(X_batch)
                    loss = criterion(predictions, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    
            val_loss /= len(val_loader.dataset)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d}/{self.config['epochs']}: Train MSE = {train_loss:.6f} | Val MSE = {val_loss:.6f}")
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["patience"]:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best model state.")
                    break
                    
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        # Plot training loss if path specified
        if loss_plot_path:
            self.plot_loss_history(history, loss_plot_path)
            
        return history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for target prices. Returns actual dollars close prices.
        Assumes features are aligned with sequence length config.
        """
        self.model.eval()
        X_scaled = self.scale_features(X)
        dataset = TimeSeriesDataset(X_scaled, np.zeros(len(X)), self.config["seq_length"])
        
        preds = np.full(len(X), np.nan)
        if len(dataset) == 0:
            return preds
            
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)
        
        scaled_preds_list = []
        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                predictions = self.model(X_batch).cpu().numpy()
                scaled_preds_list.extend(predictions)
                
        # Inverse transform scaled predictions to actual price values
        unscaled_preds = self.target_scaler.inverse_transform(np.array(scaled_preds_list).reshape(-1, 1)).flatten()
        
        # Align with the dataframe row indices
        preds[self.config["seq_length"] - 1 :] = unscaled_preds
        return preds

    def plot_loss_history(self, history: Dict[str, list], filepath: str) -> None:
        """
        Plots the training and validation loss curve and saves it as a high-quality PNG.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        plt.figure(figsize=(10, 5))
        
        plt.plot(history["train_loss"], label="Train MSE Loss", color="#4f46e5", linewidth=2.5)
        plt.plot(history["val_loss"], label="Validation MSE Loss", color="#10b981", linewidth=2.5)
        
        plt.title(f"LSTM Price Forecaster: Training & Validation Loss Curve\n(Best Val Loss: {min(history['val_loss']):.6f})", fontsize=12, fontweight="bold", pad=15)
        plt.xlabel("Epochs", fontsize=11, labelpad=10)
        plt.ylabel("Loss (MSE)", fontsize=11, labelpad=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=10, loc="upper right")
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Loss curve plot saved successfully to {filepath}")

    def save(self, dir_path: str, model_name: str = "lstm_model") -> None:
        """
        Saves weights, config, and scaler.
        """
        os.makedirs(dir_path, exist_ok=True)
        
        # Save PyTorch weights
        model_path = os.path.join(dir_path, f"{model_name}.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save Scalers
        feature_scaler_path = os.path.join(dir_path, f"{model_name}_scaler.pkl")
        target_scaler_path = os.path.join(dir_path, f"{model_name}_target_scaler.pkl")
        joblib.dump(self.scaler, feature_scaler_path)
        joblib.dump(self.target_scaler, target_scaler_path)
        
        # Save config
        config_path = os.path.join(dir_path, f"{model_name}_config.json")
        with open(config_path, "w") as f:
            json_config = self.config.copy()
            json_config["device"] = str(json_config["device"])
            json.dump(json_config, f, indent=4)
            
        logger.info(f"LSTM regression model files saved successfully in {dir_path}")

    def load(self, dir_path: str, model_name: str = "lstm_model") -> None:
        """
        Loads weights, config, and scalers.
        """
        config_path = os.path.join(dir_path, f"{model_name}_config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.get("device") != "cpu" else "cpu")
        self.model = StockLSTM(
            input_dim=self.model.lstm.input_size,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"]
        ).to(self.device)
        
        # Load weights
        model_path = os.path.join(dir_path, f"{model_name}.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load Scalers
        feature_scaler_path = os.path.join(dir_path, f"{model_name}_scaler.pkl")
        target_scaler_path = os.path.join(dir_path, f"{model_name}_target_scaler.pkl")
        self.scaler = joblib.load(feature_scaler_path)
        self.target_scaler = joblib.load(target_scaler_path)
        
        logger.info(f"LSTM regression model loaded successfully from {dir_path}")
