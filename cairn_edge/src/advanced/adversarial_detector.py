"""Adversarial patch detection for Cairn-Edge.

Threat model:
    This module is designed to flag static printed patches and stickers visible
    inside high-confidence RGB detections. It does not defend against projection
    attacks, digital video injection, compromised camera firmware, or full-scene
    adversarial perturbations.

Deployment target:
    Jetson Orin Nano, TensorRT INT8 engine, batch=1, 64x64 RGB ROI, target
    latency <10 ms per ROI. If the engine is missing or unavailable, detection
    is skipped and health is degraded rather than blocking the tracking loop.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Tuple

import cv2
import numpy as np

from .models import Detection, HealthStatus

LOGGER = logging.getLogger("cairn_edge.adversarial")


class InferenceBackend(Protocol):
    def predict_logits(self, roi_rgb_64: np.ndarray) -> np.ndarray:
        """Return logits or probabilities shaped (2,) for [real, adversarial]."""


class TensorRTAdversarialBackend:
    """TensorRT backend placeholder with explicit failure when runtime is absent.

    The hot path is designed for TensorRT, but this repository does not vendor
    TensorRT bindings. Production images should provide a backend that binds the
    serialized INT8 engine with CUDA buffers and returns two logits.
    """

    def __init__(self, engine_path: str | Path) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"adversarial TensorRT engine missing: {self.engine_path}")
        try:
            import tensorrt as trt  # type: ignore  # noqa: F401
            import pycuda.driver as cuda  # type: ignore  # noqa: F401
        except Exception as exc:  # pragma: no cover - Jetson runtime only
            raise RuntimeError("TensorRT/PyCUDA runtime unavailable") from exc
        raise NotImplementedError("bind TensorRT execution context in deployment image")

    def predict_logits(self, roi_rgb_64: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class HeuristicMockBackend:
    """Deterministic test backend; not for field deployment."""

    def predict_logits(self, roi_rgb_64: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi_rgb_64, cv2.COLOR_RGB2GRAY)
        edge_score = float(np.std(gray)) / 64.0
        adversarial_prob = float(np.clip(edge_score, 0.0, 1.0))
        return np.array([1.0 - adversarial_prob, adversarial_prob], dtype=np.float32)


class AdversarialPatchDetector:
    """Run adversarial check on high-confidence detections only."""

    def __init__(self, engine_path: str | Path, backend: Optional[InferenceBackend] = None, threshold: float = 0.8, min_confidence: float = 0.7) -> None:
        self.engine_path = Path(engine_path)
        self.threshold = threshold
        self.min_confidence = min_confidence
        self.backend = backend
        self._last_health = time.time()
        self._degraded_reason: Optional[str] = None
        if self.backend is None:
            try:
                self.backend = TensorRTAdversarialBackend(self.engine_path)
            except Exception as exc:
                LOGGER.warning("Adversarial detector disabled: %s", exc)
                self._degraded_reason = f"adversarial detector unavailable: {exc}"
                self.backend = None

    def preprocess_roi(self, frame_bgr: np.ndarray, detection: Detection) -> Optional[np.ndarray]:
        x, y, w, h = detection.bbox
        height, width = frame_bgr.shape[:2]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(width, int(x + w))
        y1 = min(height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None
        crop_bgr = frame_bgr[y0:y1, x0:x1]
        resized_bgr = cv2.resize(crop_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        arr = logits.astype(np.float32).reshape(-1)
        arr = arr - np.max(arr)
        exp = np.exp(arr)
        denom = float(np.sum(exp))
        return exp / denom if denom > 0 else np.array([1.0, 0.0], dtype=np.float32)

    def predict_roi(self, roi_rgb_64: np.ndarray) -> Tuple[bool, float]:
        if self.backend is None:
            return False, 0.0
        logits = self.backend.predict_logits(roi_rgb_64)
        probs = self._softmax(logits)
        score = float(probs[1])
        return score > self.threshold, score

    def process_detections(self, frame_bgr: np.ndarray, detections: Iterable[Detection]) -> List[Detection]:
        output: List[Detection] = []
        for detection in detections:
            if detection.confidence <= self.min_confidence:
                output.append(detection)
                continue
            roi = self.preprocess_roi(frame_bgr, detection)
            if roi is None:
                output.append(detection)
                continue
            detection.roi_crop = roi
            is_adv, score = self.predict_roi(roi)
            detection.adversarial = is_adv
            detection.adversarial_score = score
            output.append(detection)
        self._last_health = time.time()
        return output

    def health(self) -> HealthStatus:
        return HealthStatus(
            module_name="adversarial_patch_detector",
            status="degraded" if self._degraded_reason else "ok",
            last_heartbeat=self._last_health,
            degraded_reason=self._degraded_reason,
        )
