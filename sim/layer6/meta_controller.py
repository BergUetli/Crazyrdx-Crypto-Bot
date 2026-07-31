"""
meta_controller.py
Monitors all layers for drift, anomalies, and performance degradation.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

import numpy as np

from config import LOG_DIR


@dataclass
class LayerMetrics:
    """Metrics for a single layer."""
    layer_name: str
    timestamp: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    loss: float = 0.0
    auc: float = 0.0
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class SystemAlert:
    """Alert from the meta-controller."""
    timestamp: float
    severity: str  # "info", "warning", "critical"
    layer: str
    message: str
    metric_value: float
    threshold: float


class DriftDetector:
    """
    Detects distribution drift using Population Stability Index (PSI).
    """

    def __init__(self, reference_data: np.ndarray, n_bins: int = 10):
        self.reference = reference_data
        self.n_bins = n_bins
        self._compute_reference_distribution()

    def _compute_reference_distribution(self):
        """Compute reference histogram."""
        self.ref_hist, self.bin_edges = np.histogram(self.reference, bins=self.n_bins)
        self.ref_hist = self.ref_hist / len(self.reference)

    def compute_psi(self, current_data: np.ndarray) -> float:
        """
        Compute PSI between reference and current data.
        PSI > 0.25 indicates significant drift.
        """
        curr_hist, _ = np.histogram(current_data, bins=self.bin_edges)
        curr_hist = curr_hist / len(current_data)

        # Avoid division by zero
        eps = 1e-10
        psi = np.sum(
            (curr_hist - self.ref_hist) * np.log((curr_hist + eps) / (self.ref_hist + eps))
        )

        return float(psi)


class MetaController:
    """
    Watches all layers and triggers alerts when performance degrades.
    """

    def __init__(
        self,
        drift_threshold: float = 0.25,
        auc_drop_threshold: float = 0.10,
        precision_drop_threshold: float = 0.15,
        cost_ceiling_chf: float = 2.00,
    ):
        self.drift_threshold = drift_threshold
        self.auc_drop_threshold = auc_drop_threshold
        self.precision_drop_threshold = precision_drop_threshold
        self.cost_ceiling = cost_ceiling_chf

        self.layer_history: Dict[str, List[LayerMetrics]] = {}
        self.alerts: List[SystemAlert] = []
        self.drift_detectors: Dict[str, DriftDetector] = {}

        # Baseline metrics (set after initial training)
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}

    def register_drift_detector(self, name: str, reference_data: np.ndarray):
        """Register a drift detector for a data stream."""
        self.drift_detectors[name] = DriftDetector(reference_data)

    def check_drift(self, name: str, current_data: np.ndarray) -> Optional[SystemAlert]:
        """Check for drift in a data stream."""
        if name not in self.drift_detectors:
            return None

        psi = self.drift_detectors[name].compute_psi(current_data)

        if psi > self.drift_threshold:
            alert = SystemAlert(
                timestamp=time.time(),
                severity="warning",
                layer=name,
                message=f"Data drift detected: PSI={psi:.4f} > {self.drift_threshold}",
                metric_value=psi,
                threshold=self.drift_threshold,
            )
            self.alerts.append(alert)
            return alert

        return None

    def log_metrics(self, metrics: LayerMetrics):
        """Log metrics for a layer."""
        if metrics.layer_name not in self.layer_history:
            self.layer_history[metrics.layer_name] = []
        self.layer_history[metrics.layer_name].append(metrics)

    def check_performance_degradation(self, layer_name: str) -> Optional[SystemAlert]:
        """Check if a layer's performance has degraded vs baseline."""
        if layer_name not in self.baseline_metrics:
            return None

        if layer_name not in self.layer_history or len(self.layer_history[layer_name]) == 0:
            return None

        baseline = self.baseline_metrics[layer_name]
        current = self.layer_history[layer_name][-1]

        # Check AUC drop
        if "auc" in baseline and current.auc > 0:
            auc_drop = baseline["auc"] - current.auc
            if auc_drop > self.auc_drop_threshold:
                alert = SystemAlert(
                    timestamp=time.time(),
                    severity="warning",
                    layer=layer_name,
                    message=f"AUC dropped by {auc_drop:.4f}: {baseline['auc']:.4f} -> {current.auc:.4f}",
                    metric_value=current.auc,
                    threshold=baseline["auc"] - self.auc_drop_threshold,
                )
                self.alerts.append(alert)
                return alert

        # Check precision drop
        if "precision" in baseline and current.precision > 0:
            prec_drop = baseline["precision"] - current.precision
            if prec_drop > self.precision_drop_threshold:
                alert = SystemAlert(
                    timestamp=time.time(),
                    severity="warning",
                    layer=layer_name,
                    message=f"Precision dropped by {prec_drop:.4f}",
                    metric_value=current.precision,
                    threshold=baseline["precision"] - self.precision_drop_threshold,
                )
                self.alerts.append(alert)
                return alert

        return None

    def check_cost_ceiling(self, current_cost_chf: float) -> Optional[SystemAlert]:
        """Check if cost exceeds ceiling."""
        if current_cost_chf > self.cost_ceiling:
            alert = SystemAlert(
                timestamp=time.time(),
                severity="critical",
                layer="cost_tracker",
                message=f"Cost ceiling exceeded: {current_cost_chf:.4f} CHF > {self.cost_ceiling} CHF",
                metric_value=current_cost_chf,
                threshold=self.cost_ceiling,
            )
            self.alerts.append(alert)
            return alert
        return None

    def get_alerts(self, severity: Optional[str] = None, since_ts: float = 0) -> List[SystemAlert]:
        """Get alerts, optionally filtered by severity and time."""
        alerts = [a for a in self.alerts if a.timestamp > since_ts]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def save_report(self, path: Path):
        """Save meta-controller report."""
        report = {
            "timestamp": time.time(),
            "alerts": [asdict(a) for a in self.alerts[-100:]],  # last 100
            "layer_metrics": {
                name: [asdict(m) for m in metrics[-10:]]  # last 10 per layer
                for name, metrics in self.layer_history.items()
            },
            "baseline_metrics": self.baseline_metrics,
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def set_baseline(self, layer_name: str, metrics: Dict[str, float]):
        """Set baseline metrics for a layer."""
        self.baseline_metrics[layer_name] = metrics
