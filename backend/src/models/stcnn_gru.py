"""
ST-CNN-GRU: Spatiotemporal CNN-GRU with Cross-Station Spatial Attention for PM2.5 Forecasting.

Scientific Architecture:
1. Multi-Scale Temporal CNN:
   Extracts both short-term rapid variations (kernel 3) and diurnal patterns (kernel 7)
   for each monitoring station across historical lookback time.
2. Cross-Station Spatial Graph Attention:
   Combines representations across all 12 Beijing monitoring stations using:
   - Pairwise inter-station Haversine geographic distances (calibrated spatial proximity prior).
   - Dynamic multi-station feature attention and graph message passing.
   - Genuine cross-station information propagation (perturbing station j directly affects station i).
3. Stacked Recurrent GRU:
   Models long-term sequential atmospheric dynamics and multiday stagnation cycles.
4. Multi-Horizon Prediction Heads:
   Horizon-specific projections for 1h, 6h, 12h, 24h, 48h, and 72h lead times.
"""
import math
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.models.lstm import LSTMModel
from src.config.settings import MODELS_DIR, STATION_COORDS, EXPECTED_STATIONS


def compute_distance_matrix() -> torch.Tensor:
    """Compute normalized pairwise Haversine distances between all 12 stations."""
    stations = [s for s in EXPECTED_STATIONS if s in STATION_COORDS]
    n = len(stations)
    dist = torch.zeros(n, n)
    for i, s1 in enumerate(stations):
        lat1, lon1 = STATION_COORDS[s1]
        for j, s2 in enumerate(stations):
            lat2, lon2 = STATION_COORDS[s2]
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
            dist[i, j] = 2 * math.asin(math.sqrt(max(0.0, a))) * 6371.0  # distance in km
    max_d = dist.max()
    if max_d > 0:
        dist = dist / max_d
    return dist


def compute_adjacency_matrix(gamma: float = 2.0) -> torch.Tensor:
    """Compute row-normalized distance-based spatial adjacency matrix."""
    dist = compute_distance_matrix()
    adj = torch.exp(-dist * gamma)
    adj = adj / adj.sum(dim=-1, keepdim=True)
    return adj


class MultiScaleCNN(nn.Module):
    """Multi-scale 1D CNN for temporal feature extraction over lookback sequences."""

    def __init__(self, input_channels: int, out_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.project = nn.Conv1d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_total, channels, seq_len)
        c3 = self.conv3(x)
        c7 = self.conv7(x)
        combined = torch.cat([c3, c7], dim=1)
        return self.project(combined)


class CrossStationSpatialAttention(nn.Module):
    """
    Cross-station spatial message passing and feature attention layer.
    Propagates information across the 12 monitoring stations weighted by
    both geographic Haversine distance priors and learned dynamic correlation.
    """

    def __init__(self, channels: int, n_stations: int = 12):
        super().__init__()
        self.channels = channels
        self.n_stations = n_stations
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.spatial_gcn = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, C)
        # adj: (N, N) row-normalized spatial proximity matrix
        B, T, N, C = h.shape

        # 1. Geographic Graph Message Passing: A * H * W
        # adj: (1, 1, N, N) x h: (B, T, N, C) -> (B, T, N, C)
        h_gcn = torch.matmul(adj.unsqueeze(0).unsqueeze(0), h)
        h_gcn = self.spatial_gcn(h_gcn)

        # 2. Dynamic Cross-Station Attention
        Q = self.q(h)  # (B, T, N, C)
        K = self.k(h)  # (B, T, N, C)
        V = self.v(h)  # (B, T, N, C)

        # Cross-station dot product attention over station dimension N
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (C ** 0.5)  # (B, T, N, N)
        # Incorporate distance matrix as structural spatial prior
        scores = scores + torch.log(adj.unsqueeze(0).unsqueeze(0) + 1e-6)
        attn = F.softmax(scores, dim=-1)
        h_attn = self.out_proj(torch.matmul(attn, V))

        # Residual connection + LayerNorm
        h_out = self.norm(h + h_gcn + h_attn)
        return h_out


class STCNNGRUNetwork(nn.Module):
    """
    Spatiotemporal CNN-GRU Network with genuine cross-station spatial learning.
    Input shapes supported:
    - 4D Tensor: (B, T, N, F) — genuine multi-station spatiotemporal grid
    - 3D Tensor: (B, T, F) — single station or legacy unit test, automatically expanded
    Output shapes:
    - (B, N, n_horizons) for multi-station output
    - (B, n_horizons) when target_station_idx is specified or for 3D input
    """

    def __init__(self, input_size: int, cnn_channels: int = 64,
                 gru_hidden: int = 128, gru_layers: int = 2,
                 dropout: float = 0.2, n_horizons: int = 6,
                 n_stations: int = 12, use_spatial: bool = True):
        super().__init__()
        self.input_size = input_size
        self.cnn_channels = cnn_channels
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.n_stations = n_stations
        self.n_horizons = n_horizons
        self.use_spatial = use_spatial

        self.temporal_cnn = MultiScaleCNN(input_size, cnn_channels, dropout)

        # Haversine distance-based adjacency matrix buffer
        adj_mat = compute_adjacency_matrix()
        self.register_buffer("proximity", adj_mat)

        # Genuine Cross-Station Spatial Attention Layer
        self.spatial_attention = CrossStationSpatialAttention(cnn_channels, n_stations)

        # Stacked Recurrent GRU over time
        self.gru = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Multi-horizon linear prediction heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(gru_hidden, gru_hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(gru_hidden // 2, 1),
            )
            for _ in range(n_horizons)
        ])

    def forward(self, x: torch.Tensor, target_station_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.
        x: (B, T, N, F) or (B, T, F)
        target_station_idx: optional integer 0..11 to select single station output.
        """
        was_3d = False
        if x.dim() == 3:
            was_3d = True
            # Expand (B, T, F) -> (B, T, N, F)
            B, T, F_in = x.shape
            x = x.unsqueeze(2).repeat(1, 1, self.n_stations, 1)

        B, T, N, F_in = x.shape

        # 1. Temporal Feature Extraction: (B*N, F_in, T)
        x_flat = x.permute(0, 2, 3, 1).reshape(B * N, F_in, T)
        cnn_out = self.temporal_cnn(x_flat)  # (B*N, cnn_channels, T)
        # Reshape to (B, T, N, cnn_channels)
        cnn_out = cnn_out.view(B, N, self.cnn_channels, T).permute(0, 3, 1, 2)

        # 2. Genuine Cross-Station Spatial Attention & Graph Message Passing
        if self.use_spatial:
            h_spatial = self.spatial_attention(cnn_out, self.proximity)
        else:
            h_spatial = cnn_out

        # 3. Stacked Temporal GRU
        # Reshape to (B*N, T, cnn_channels)
        gru_in = h_spatial.permute(0, 2, 1, 3).reshape(B * N, T, self.cnn_channels)
        gru_out, _ = self.gru(gru_in)  # (B*N, T, gru_hidden)
        last_hidden = self.dropout(gru_out[:, -1, :])  # (B*N, gru_hidden)
        last_hidden = last_hidden.view(B, N, self.gru_hidden)  # (B, N, gru_hidden)

        # 4. Multi-Horizon Prediction Heads
        # Output shape: (B, N, n_horizons)
        preds_all = torch.cat([head(last_hidden) for head in self.output_heads], dim=-1)

        if target_station_idx is not None:
            return preds_all[:, target_station_idx, :]
        elif was_3d:
            return preds_all[:, 0, :]
        else:
            return preds_all


class STCNNGRUModel(LSTMModel):
    """
    ST-CNN-GRU Model wrapper with genuine cross-station spatial learning.
    Integrates multi-station spatiotemporal training and inference.
    """

    def __init__(self, input_size: int, cnn_channels: int = 64,
                 gru_hidden: int = 128, gru_layers: int = 2,
                 dropout: float = 0.2, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 64,
                 max_epochs: int = 100, patience: int = 15,
                 grad_clip: float = 1.0, seq_len: int = 24,
                 horizons: List[int] = [1, 6, 12, 24, 48, 72],
                 n_stations: int = 12, use_spatial: bool = True,
                 random_state: int = 42, device: Optional[str] = None):

        super().__init__(
            input_size=input_size, hidden_size=gru_hidden,
            num_layers=gru_layers, dropout=dropout,
            learning_rate=learning_rate, weight_decay=weight_decay,
            batch_size=batch_size, max_epochs=max_epochs,
            patience=patience, grad_clip=grad_clip,
            seq_len=seq_len, horizons=horizons,
            random_state=random_state, device=device,
        )
        self.name = "stcnn_gru"
        self.n_stations = n_stations
        self.use_spatial = use_spatial

        self.network = STCNNGRUNetwork(
            input_size=input_size, cnn_channels=cnn_channels,
            gru_hidden=gru_hidden, gru_layers=gru_layers,
            dropout=dropout, n_horizons=len(horizons),
            n_stations=n_stations, use_spatial=use_spatial,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )

    @staticmethod
    def build_spatiotemporal_tensors(df: pd.DataFrame, feature_cols: List[str],
                                     target_cols: List[str], seq_len: int = 24,
                                     stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct genuine 4D sliding-window spatiotemporal tensors (N_windows, seq_len, 12, n_features)
        and 3D targets (N_windows, 12, n_horizons) in deterministic station order.
        Strictly prevents information leakage across chronological boundaries.
        """
        # Ensure stations are in canonical order
        st_map = {st: i for i, st in enumerate(EXPECTED_STATIONS)}
        df_copy = df.copy()
        df_copy["st_idx"] = df_copy["station"].map(st_map)
        df_sorted = df_copy.sort_values(["datetime", "st_idx"]).reset_index(drop=True)

        n_ts = df_sorted["datetime"].nunique()
        n_st = len(EXPECTED_STATIONS)
        n_feat = len(feature_cols)
        n_targets = len(target_cols)

        # 3D full grids: (N_timestamps, 12, n_features)
        feat_3d = df_sorted[feature_cols].fillna(0).values.reshape(n_ts, n_st, n_feat).astype(np.float32)
        targ_3d = df_sorted[target_cols].fillna(0).values.reshape(n_ts, n_st, n_targets).astype(np.float32)

        windows_X, windows_y = [], []
        for i in range(0, n_ts - seq_len + 1, stride):
            # Causal lookback window: past seq_len timestamps up to current step
            windows_X.append(feat_3d[i : i + seq_len])
            # Target at the last timestamp of the window
            windows_y.append(targ_3d[i + seq_len - 1])

        X_4d = np.array(windows_X, dtype=np.float32)
        y_3d = np.array(windows_y, dtype=np.float32)
        return X_4d, y_3d

    def fit_spatiotemporal(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
                           feature_cols: Optional[List[str]] = None,
                           target_cols: Optional[List[str]] = None,
                           checkpoint_path: Optional[Path] = None,
                           stride: int = 1) -> "STCNNGRUModel":
        """
        Fit ST-CNN-GRU using genuine multi-station 4D sliding window tensors.
        """
        if feature_cols is None:
            target_cols = [f"target_h{h}" for h in self.horizons]
            exclude = ["datetime", "station"] + target_cols
            feature_cols = [c for c in train_df.columns if c not in exclude]

        logger.info("Constructing genuine 4D spatiotemporal tensors for multi-station training...")
        X_tr, y_tr = self.build_spatiotemporal_tensors(train_df, feature_cols, target_cols, self.seq_len, stride=stride)
        logger.info(f"Training spatiotemporal tensor: X={X_tr.shape}, y={y_tr.shape}")

        X_va, y_va = None, None
        if val_df is not None:
            X_va, y_va = self.build_spatiotemporal_tensors(val_df, feature_cols, target_cols, self.seq_len, stride=max(1, stride))
            logger.info(f"Validation spatiotemporal tensor: X={X_va.shape}, y={y_va.shape}")

        train_ds = torch.utils.data.TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_va is not None:
            val_ds = torch.utils.data.TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            self.network.train()
            total_loss = 0.0

            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)  # (B, 12, 6)

                self.optimizer.zero_grad()
                preds = self.network(bx)  # (B, 12, 6)
                loss = self.criterion(preds, by)
                loss.backward()

                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)

                self.optimizer.step()
                total_loss += loss.item() * len(bx)

            avg_train_loss = total_loss / len(train_ds)
            self.train_history["train_loss"].append(avg_train_loss)

            # Validation
            val_loss = avg_train_loss
            if val_loader is not None:
                self.network.eval()
                v_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        p = self.network(vx)
                        v_loss += self.criterion(p, vy).item() * len(vx)
                val_loss = v_loss / len(val_ds)
                self.train_history["val_loss"].append(val_loss)

            self.scheduler.step()

            if val_loss < best_loss:
                best_loss = val_loss
                self.best_val_loss = best_loss
                self.best_epoch = epoch
                patience_counter = 0
                if checkpoint_path:
                    self.save(checkpoint_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == self.max_epochs or patience_counter == 0:
                logger.info(f"Epoch {epoch:02d}/{self.max_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f} (ep {self.best_epoch})")

            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
                station: Optional[str] = None) -> np.ndarray:
        """
        Generate multi-horizon forecasts.
        Supports 4D tensors, 3D sequences, DataFrames, and 2D arrays.
        """
        self.network.eval()
        st_idx = None
        if station and station in EXPECTED_STATIONS:
            st_idx = EXPECTED_STATIONS.index(station)

        if isinstance(X, pd.DataFrame):
            # Check if DataFrame contains station and datetime for full spatiotemporal reconstruction
            if "station" in X.columns and "datetime" in X.columns:
                target_cols = [f"target_h{h}" for h in self.horizons if f"target_h{h}" in X.columns]
                exclude = ["datetime", "station"] + target_cols
                f_cols = [c for c in X.columns if c not in exclude]
                X_4d, _ = self.build_spatiotemporal_tensors(X, f_cols, target_cols or [f_cols[0]], self.seq_len)
                with torch.no_grad():
                    t_x = torch.tensor(X_4d, dtype=torch.float32, device=self.device)
                    out = self.network(t_x, target_station_idx=st_idx).cpu().numpy()
                return out
            else:
                X = X.fillna(0).values.astype(np.float32)

        if isinstance(X, np.ndarray):
            if X.ndim == 4:
                # (B, T, N, F)
                with torch.no_grad():
                    t_x = torch.tensor(X, dtype=torch.float32, device=self.device)
                    out = self.network(t_x, target_station_idx=st_idx).cpu().numpy()
                return out
            elif X.ndim == 3:
                # (B, T, F)
                with torch.no_grad():
                    t_x = torch.tensor(X, dtype=torch.float32, device=self.device)
                    out = self.network(t_x).cpu().numpy()
                return out
            elif X.ndim == 2:
                # 2D array (N_samples, F): create sliding sequences
                n_samples, n_feat = X.shape
                if n_samples < self.seq_len:
                    padding = np.zeros((self.seq_len - n_samples, n_feat), dtype=np.float32)
                    X_seq = np.vstack([padding, X])
                    seqs = np.expand_dims(X_seq, axis=0)
                else:
                    seqs = []
                    for i in range(n_samples - self.seq_len + 1):
                        seqs.append(X[i : i + self.seq_len])
                    seqs = np.array(seqs, dtype=np.float32)
                with torch.no_grad():
                    t_x = torch.tensor(seqs, dtype=torch.float32, device=self.device)
                    out = self.network(t_x).cpu().numpy()
                return out

        raise ValueError(f"Unsupported input type or shape for STCNNGRUModel.predict: {type(X)}")

    def save(self, path: Optional[Path] = None) -> Path:
        save_path = path or (MODELS_DIR / "stcnn_gru_model.pt")
        torch.save({
            "network_state": self.network.state_dict(),
            "train_history": self.train_history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "config": {
                "input_size": self.network.input_size,
                "cnn_channels": self.network.cnn_channels,
                "gru_hidden": self.network.gru_hidden,
                "gru_layers": self.network.gru_layers,
                "seq_len": self.seq_len,
                "horizons": self.horizons,
                "n_stations": self.n_stations,
                "use_spatial": self.use_spatial,
            }
        }, save_path)
        logger.info(f"STCNNGRUModel saved to {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Optional[Path] = None, device: Optional[str] = None) -> "STCNNGRUModel":
        load_path = path or (MODELS_DIR / "stcnn_gru_model.pt")
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model = cls(
            input_size=cfg["input_size"],
            cnn_channels=cfg.get("cnn_channels", 64),
            gru_hidden=cfg.get("gru_hidden", 128),
            gru_layers=cfg.get("gru_layers", 2),
            seq_len=cfg.get("seq_len", 24),
            horizons=cfg.get("horizons", [1, 6, 12, 24, 48, 72]),
            n_stations=cfg.get("n_stations", 12),
            use_spatial=cfg.get("use_spatial", True),
            device=device,
        )
        model.network.load_state_dict(checkpoint["network_state"])
        model.train_history = checkpoint.get("train_history", {})
        model.best_val_loss = checkpoint.get("best_val_loss", 0.0)
        model.best_epoch = checkpoint.get("best_epoch", 1)
        model.fitted = True
        return model
