"""
dataset.py
Loads feature vectors from SQLite and prepares them for autoencoder training.
"""

import json
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pathlib import Path

from config import DB_FEATURES, DB_RAW_QUOTES


# Feature names in order (35 features, excluding ts and pair)
FEATURE_NAMES = [
    "price", "price_impact_pct",
    "spread_bps", "spread_acceleration", "spread_jerk",
    "rolling_spread_mean_30s", "rolling_spread_std_30s",
    "rolling_spread_max_30s", "rolling_spread_skew_30s",
    "rolling_spread_kurtosis_30s",
    "price_roc_5s", "price_roc_30s", "price_roc_60s",
    "price_volatility_30s", "price_volatility_300s",
    "pool_liquidity_usd", "pool_depth_ratio",
    "pool_volume_24h", "pool_fee_tier",
    "cex_dex_spread_bps", "cex_price_lead_ms", "cross_pair_correlation",
    "solana_slot_time_ms", "network_congestion_score", "compute_unit_price",
    "time_of_day_sin", "time_of_day_cos", "day_of_week",
    "seconds_since_last_spike", "is_weekend",
    "spread_bps_lag_1", "spread_bps_lag_2", "spread_bps_lag_3",
    "price_lag_1", "price_lag_2",
]


class FeatureDataset(Dataset):
    """PyTorch dataset for feature vectors."""

    def __init__(self, features: np.ndarray):
        """
        Args:
            features: (N, 35) numpy array of feature vectors
        """
        self.features = torch.FloatTensor(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def load_features_from_db(
    pair: Optional[str] = None,
    since_ts: float = 0,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load feature vectors from SQLite.

    Returns:
        features: (N, 35) numpy array
        pairs: list of pair names corresponding to each row
    """
    conn = sqlite3.connect(str(DB_FEATURES))
    conn.row_factory = sqlite3.Row

    query = "SELECT pair, features_json FROM feature_vectors WHERE ts > ?"
    params = [since_ts]

    if pair:
        query += " AND pair = ?"
        params.append(pair)

    query += " ORDER BY ts"

    if limit:
        query += f" LIMIT {limit}"

    cursor = conn.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    features = []
    pairs = []

    for row in rows:
        fv = json.loads(row["features_json"])
        # Extract features in canonical order
        vec = [fv.get(name, 0.0) for name in FEATURE_NAMES]
        features.append(vec)
        pairs.append(row["pair"])

    return np.array(features, dtype=np.float32), pairs


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize features to zero mean, unit variance.

    Returns:
        normalized: (N, 35) normalized features
        stats: dict with mean and std per feature
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0

    normalized = (features - mean) / std

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    return normalized, stats


def denormalize_features(normalized: np.ndarray, stats: dict) -> np.ndarray:
    """Reverse normalization."""
    mean = np.array(stats["mean"])
    std = np.array(stats["std"])
    return normalized * std + mean


def create_dataloader(
    pair: Optional[str] = None,
    since_ts: float = 0,
    batch_size: int = 32,
    shuffle: bool = True,
    normalize: bool = True,
) -> Tuple[DataLoader, Optional[dict]]:
    """
    Create a DataLoader from database features.

    Returns:
        dataloader: PyTorch DataLoader
        norm_stats: normalization stats if normalize=True, else None
    """
    features, _ = load_features_from_db(pair, since_ts)

    if len(features) == 0:
        raise ValueError("No features found in database")

    norm_stats = None
    if normalize:
        features, norm_stats = normalize_features(features)

    dataset = FeatureDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, norm_stats
