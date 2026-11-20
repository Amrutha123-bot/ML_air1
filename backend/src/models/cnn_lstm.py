"""CNN-LSTM model for PM2.5 multi-horizon forecasting."""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger
from src.models.lstm import LSTMModel
from src.config.settings import MODELS_DIR


class CNNLSTMNetwork(nn.Module):
    """
    CNN-LSTM: 1D Convolutional feature extraction + LSTM temporal modeling.
    Input: (batch, seq_len, n_features)
    Output: (batch, n_horizons)
    """

    def __init__(self, input_size: int, cnn_channels: int = 64,
                 cnn_kernel: int = 3, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.2,
                 n_horizons: int = 6):
        super().__init__()
        self.n_horizons = n_horizons

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=cnn_kernel, padding=cnn_kernel // 2),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, n_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> permute -> (batch, features, seq_len)
        x_perm = x.permute(0, 2, 1)
        cnn_out = self.cnn(x_perm)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, seq_len, cnn_channels*2)
        lstm_out, _ = self.lstm(cnn_out)
        last = self.dropout(lstm_out[:, -1, :])
        return self.fc(last)


class CNNLSTMModel(LSTMModel):
    """CNN-LSTM model inheriting LSTM training infrastructure."""

    def __init__(self, input_size: int, cnn_channels: int = 64,
                 cnn_kernel: int = 3, lstm_hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.2,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 64, max_epochs: int = 100,
                 patience: int = 15, grad_clip: float = 1.0,
                 seq_len: int = 24, horizons: List[int] = [1, 6, 12, 24, 48, 72],
                 random_state: int = 42, device: Optional[str] = None):

        super().__init__(
            input_size=input_size, hidden_size=lstm_hidden,
            num_layers=lstm_layers, dropout=dropout,
            learning_rate=learning_rate, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, grad_clip=grad_clip,
            seq_len=seq_len, horizons=horizons,
            random_state=random_state, device=device,
        )
        self.name = "cnn_lstm"

        self.network = CNNLSTMNetwork(
            input_size=input_size, cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel, lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers, dropout=dropout,
            n_horizons=len(horizons),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, min_lr=1e-6
        )

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (MODELS_DIR / "cnn_lstm_model.pt")
        torch.save({
            "network_state": self.network.state_dict(),
            "train_history": self.train_history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": {
                "input_size": self.network.cnn[0].in_channels,
                "cnn_channels": self.network.cnn[0].out_channels,
                "lstm_hidden": self.network.lstm.hidden_size,
                "lstm_layers": self.network.lstm.num_layers,
                "seq_len": self.seq_len,
                "horizons": self.horizons,
            }
        }, save_path)
        logger.info(f"CNNLSTMModel saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None, device: Optional[str] = None) -> "CNNLSTMModel":
        load_path = path or (MODELS_DIR / "cnn_lstm_model.pt")
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model = cls(
            input_size=cfg["input_size"], cnn_channels=cfg["cnn_channels"],
            lstm_hidden=cfg["lstm_hidden"], lstm_layers=cfg["lstm_layers"],
            seq_len=cfg["seq_len"], horizons=cfg["horizons"], device=device,
        )
        model.network.load_state_dict(checkpoint["network_state"])
        model.train_history = checkpoint.get("train_history", {})
        model.best_val_loss = checkpoint.get("best_val_loss", 0.0)
        model.fitted = True
        return model
