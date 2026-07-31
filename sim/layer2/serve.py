"""
serve.py
ONNX export and fast inference for the trained autoencoder.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch

from config import MODEL_DIR
from layer2.autoencoder import get_model
from layer2.dataset import FEATURE_NAMES, normalize_features, denormalize_features


class AutoencoderServer:
    """
    Serves the trained autoencoder for inference.
    Handles ONNX export if available, falls back to PyTorch.
    """

    def __init__(
        self,
        model_type: str = "standard",
        latent_dim: int = 16,
        checkpoint: str = "best",
        device: Optional[str] = None,
    ):
        self.model_type = model_type
        self.latent_dim = latent_dim
        self.checkpoint = checkpoint

        # Device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model
        self.model = get_model(model_type, input_dim=35, latent_dim=latent_dim)
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats
        self.norm_stats = self._load_norm_stats()

        # ONNX session (lazy init)
        self._onnx_session = None
        self._onnx_available = False

    def _load_checkpoint(self):
        path = MODEL_DIR / f"autoencoder_{self.model_type}_{self.checkpoint}.pt"
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _load_norm_stats(self) -> Optional[dict]:
        path = MODEL_DIR / f"autoencoder_{self.model_type}_norm_stats.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    def export_onnx(self, output_path: Optional[Path] = None) -> Path:
        """Export model to ONNX format."""
        if output_path is None:
            output_path = MODEL_DIR / f"autoencoder_{self.model_type}_{self.checkpoint}.onnx"

        dummy_input = torch.randn(1, 35).to(self.device)

        torch.onnx.export(
            self.model.encoder,  # Just the encoder for latent extraction
            dummy_input,
            str(output_path),
            input_names=["features"],
            output_names=["latent"],
            dynamic_axes={"features": {0: "batch_size"}, "latent": {0: "batch_size"}},
            opset_version=17,
        )

        return output_path

    def init_onnx(self, onnx_path: Optional[Path] = None):
        """Initialize ONNX runtime session."""
        try:
            import onnxruntime as ort

            if onnx_path is None:
                onnx_path = MODEL_DIR / f"autoencoder_{self.model_type}_{self.checkpoint}.onnx"

            if not onnx_path.exists():
                onnx_path = self.export_onnx(onnx_path)

            # Use CoreML execution provider on Mac for M-series acceleration
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            self._onnx_session = ort.InferenceSession(str(onnx_path), providers=providers)
            self._onnx_available = True

        except ImportError:
            self._onnx_available = False

    def encode(
        self,
        features: np.ndarray,
        normalize: bool = True,
        use_onnx: bool = True,
    ) -> np.ndarray:
        """
        Extract latent representation from features.

        Args:
            features: (N, 35) numpy array
            normalize: apply normalization using stored stats
            use_onnx: use ONNX runtime if available

        Returns:
            latent: (N, latent_dim) numpy array
        """
        if normalize and self.norm_stats is not None:
            features = (features - np.array(self.norm_stats["mean"])) / np.array(self.norm_stats["std"])

        if use_onnx and self._onnx_available and self._onnx_session is not None:
            return self._encode_onnx(features)
        return self._encode_torch(features)

    def _encode_torch(self, features: np.ndarray) -> np.ndarray:
        """PyTorch inference."""
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self.device)
            latent = self.model.get_latent(x)
            return latent.cpu().numpy()

    def _encode_onnx(self, features: np.ndarray) -> np.ndarray:
        """ONNX runtime inference."""
        input_name = self._onnx_session.get_inputs()[0].name
        output_name = self._onnx_session.get_outputs()[0].name
        result = self._onnx_session.run(
            [output_name],
            {input_name: features.astype(np.float32)}
        )
        return result[0]

    def encode_single(
        self,
        feature_dict: Dict[str, Any],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a single feature vector from dict format.

        Args:
            feature_dict: dict with feature names as keys

        Returns:
            latent: (latent_dim,) numpy array
        """
        # Convert dict to ordered array
        vec = np.array([feature_dict.get(name, 0.0) for name in FEATURE_NAMES])
        vec = vec.reshape(1, -1)  # (1, 35)
        latent = self.encode(vec, normalize=normalize)
        return latent[0]  # (latent_dim,)

    def benchmark(self, n_runs: int = 1000) -> Dict[str, float]:
        """Benchmark inference speed."""
        dummy = np.random.randn(1, 35).astype(np.float32)

        # Warmup
        for _ in range(10):
            self.encode(dummy, normalize=False)

        # PyTorch benchmark
        start = time.perf_counter()
        for _ in range(n_runs):
            self._encode_torch(dummy)
        torch_time = (time.perf_counter() - start) / n_runs * 1000

        # ONNX benchmark
        onnx_time = -1.0
        if self._onnx_available:
            for _ in range(10):
                self._encode_onnx(dummy)
            start = time.perf_counter()
            for _ in range(n_runs):
                self._encode_onnx(dummy)
            onnx_time = (time.perf_counter() - start) / n_runs * 1000

        return {
            "pytorch_ms": round(torch_time, 4),
            "onnx_ms": round(onnx_time, 4) if onnx_time > 0 else None,
            "speedup": round(torch_time / onnx_time, 2) if onnx_time > 0 else None,
        }


def create_server(
    model_type: str = "standard",
    latent_dim: int = 16,
    checkpoint: str = "best",
    init_onnx: bool = True,
) -> AutoencoderServer:
    """Factory function."""
    server = AutoencoderServer(
        model_type=model_type,
        latent_dim=latent_dim,
        checkpoint=checkpoint,
    )
    if init_onnx:
        server.init_onnx()
    return server
