"""
dl_data_pipeline.py
Prepares historical feature data for deep learning model training.
"""

import json
import sqlite3
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import DATA_DIR
from layer1.historical_feature_engine import get_historical_features, HistoricalFeatureEngine

DB_HIST_FEATURES = DATA_DIR / "historical_features.db"


class SequenceDataset(Dataset):
    """
    PyTorch dataset for sequence modeling.
    Each sample is a sequence of feature vectors with a target.
    """

    def __init__(
        self,
        sequences: np.ndarray,      # (N, seq_len, n_features)
        targets: np.ndarray,        # (N, n_targets)
        pair_ids: np.ndarray,       # (N,) pair identifier for multi-pair training
    ):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.pair_ids = torch.LongTensor(pair_ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "target": self.targets[idx],
            "pair_id": self.pair_ids[idx],
        }


class HistoricalDataPipeline:
    """
    Loads historical features and prepares them for DL training.
    """

    # Feature names in canonical order (from HistoricalFeatureEngine)
    FEATURE_NAMES = HistoricalFeatureEngine.FEATURE_NAMES

    # Features to actually use (exclude raw OHLCV, keep derived features)
    INPUT_FEATURES = [
        "price_roc_5m", "price_roc_15m", "price_roc_30m", "price_roc_1h", "price_roc_4h",
        "volatility_15m", "volatility_1h", "volatility_4h", "volatility_1d",
        "hl_range_pct", "hl_range_avg_15m", "hl_range_avg_1h",
        "body_pct", "upper_wick_pct", "lower_wick_pct",
        "volume_roc_15m", "volume_roc_1h", "volume_sma_ratio", "volume_weighted_price",
        "sma_5", "sma_20", "sma_50", "price_vs_sma_20", "sma_cross_5_20", "sma_cross_20_50",
        "returns_skew_1h", "returns_kurtosis_1h", "autocorrelation_1h",
        "hour_of_day_sin", "hour_of_day_cos", "day_of_week", "is_weekend",
        "close_lag_1", "close_lag_2", "close_lag_3", "volume_lag_1",
        "returns_lag_1", "returns_lag_2",
    ]

    # Target: future returns at different horizons
    TARGET_HORIZONS = [1, 3, 6, 12]  # 5m, 15m, 30m, 1h ahead

    def __init__(
        self,
        pairs: List[str],
        seq_len: int = 24,           # 24 candles = 2 hours
        forecast_horizon: int = 12,  # 12 candles = 1 hour ahead
        normalize: bool = True,
    ):
        self.pairs = pairs
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.normalize = normalize

        self.pair_to_id = {p: i for i, p in enumerate(pairs)}
        self.norm_stats: Optional[dict] = None

    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load features for all pairs and create sequences.

        Returns:
            sequences: (N, seq_len, n_input_features)
            targets: (N, n_horizons)
            pair_ids: (N,)
        """
        all_sequences = []
        all_targets = []
        all_pair_ids = []

        for pair in self.pairs:
            print(f"Loading {pair}...")
            features = get_historical_features(pair)

            if len(features) < self.seq_len + self.forecast_horizon + 50:
                print(f"  Skipping {pair}: insufficient data ({len(features)} candles)")
                continue

            # Extract feature matrix
            n = len(features)
            feature_matrix = np.zeros((n, len(self.INPUT_FEATURES)))

            for i, f in enumerate(features):
                fv = f["features"]
                for j, name in enumerate(self.INPUT_FEATURES):
                    feature_matrix[i, j] = fv.get(name, 0.0)

            # Create sequences
            for i in range(self.seq_len, n - self.forecast_horizon):
                seq = feature_matrix[i - self.seq_len:i]

                # Target: future returns
                current_close = features[i]["features"]["close"]
                future_close = features[i + self.forecast_horizon]["features"]["close"]

                if current_close > 0:
                    future_return = (future_close - current_close) / current_close
                else:
                    future_return = 0.0

                # Also compute intermediate returns for multi-horizon prediction
                horizon_returns = []
                for h in self.TARGET_HORIZONS:
                    if i + h < n:
                        fc = features[i + h]["features"]["close"]
                        ret = (fc - current_close) / current_close if current_close > 0 else 0.0
                        horizon_returns.append(ret)
                    else:
                        horizon_returns.append(0.0)

                all_sequences.append(seq)
                all_targets.append(horizon_returns)
                all_pair_ids.append(self.pair_to_id[pair])

        sequences = np.array(all_sequences, dtype=np.float32)
        targets = np.array(all_targets, dtype=np.float32)
        pair_ids = np.array(all_pair_ids, dtype=np.int64)

        print(f"Total sequences: {len(sequences)}")
        print(f"Sequence shape: {sequences.shape}")
        print(f"Target shape: {targets.shape}")

        if self.normalize:
            sequences, self.norm_stats = self._normalize(sequences)

        return sequences, targets, pair_ids

    def _normalize(self, sequences: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Normalize sequences to zero mean, unit variance per feature."""
        # Reshape to (N*seq_len, n_features) for per-feature normalization
        n, seq_len, n_feat = sequences.shape
        flat = sequences.reshape(-1, n_feat)

        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        std[std == 0] = 1.0

        normalized = (flat - mean) / std
        normalized = normalized.reshape(n, seq_len, n_feat)

        stats = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

        return normalized, stats

    def create_dataloaders(
        self,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders with time-series split.

        Returns:
            train_loader, val_loader, test_loader
        """
        sequences, targets, pair_ids = self.load_all_data()

        n = len(sequences)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Time-series split
        train_seq = sequences[:n_train]
        train_tgt = targets[:n_train]
        train_pid = pair_ids[:n_train]

        val_seq = sequences[n_train:n_train + n_val]
        val_tgt = targets[n_train:n_train + n_val]
        val_pid = pair_ids[n_train:n_train + n_val]

        test_seq = sequences[n_train + n_val:]
        test_tgt = targets[n_train + n_val:]
        test_pid = pair_ids[n_train + n_val:]

        train_dataset = SequenceDataset(train_seq, train_tgt, train_pid)
        val_dataset = SequenceDataset(val_seq, val_tgt, val_pid)
        test_dataset = SequenceDataset(test_seq, test_tgt, test_pid)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def save_norm_stats(self, path: Path):
        """Save normalization stats for inference."""
        if self.norm_stats:
            with open(path, "w") as f:
                json.dump(self.norm_stats, f)

    def load_norm_stats(self, path: Path):
        """Load normalization stats."""
        with open(path) as f:
            self.norm_stats = json.load(f)


def prepare_training_data(
    pairs: List[str],
    seq_len: int = 24,
    forecast_horizon: int = 12,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader, HistoricalDataPipeline]:
    """
    High-level function to prepare all training data.

    Returns:
        train_loader, val_loader, test_loader, pipeline
    """
    pipeline = HistoricalDataPipeline(
        pairs=pairs,
        seq_len=seq_len,
        forecast_horizon=forecast_horizon,
        normalize=True,
    )

    train_loader, val_loader, test_loader = pipeline.create_dataloaders(
        batch_size=batch_size,
    )

    # Save normalization stats
    norm_path = DATA_DIR / "dl_norm_stats.json"
    pipeline.save_norm_stats(norm_path)
    print(f"Normalization stats saved to {norm_path}")

    return train_loader, val_loader, test_loader, pipeline
