"""Continual-learning hard example collection for Cairn-Edge.

The edge node only collects examples and verifies operator signatures. Training is
performed off-device. The hot path enqueues work and returns quickly; image and
JSONL writes are handled by a background thread.
"""
from __future__ import annotations

import base64
import json
import logging
import queue
import shutil
import time
from pathlib import Path
from threading import Event, Thread
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .models import Detection, HardExample, HealthStatus

LOGGER = logging.getLogger("cairn_edge.continual_learning")


class OperatorSignature(BaseModel):
    operator_id: str
    label: str
    signature: str


class HardExampleCollectorConfig(BaseModel):
    enabled: bool = True
    root_dir: str = "data/hard_examples"
    metadata_jsonl: str = "data/hard_examples/metadata.jsonl"
    max_examples: int = 5000
    quorum_size: int = 2
    min_free_bytes: int = 256 * 1024 * 1024
    uncertain_min_confidence: float = 0.3
    uncertain_max_confidence: float = 0.7
    public_keys_dir: str = "keys/operators"
    queue_size: int = 256


class SignatureVerifier:
    """Verify Ed25519 operator signatures.

    Message format: JSON bytes of {operator_id, label, detection_dict} sorted by
    key. Public keys are expected as PEM files named <operator_id>.pub.
    """

    def __init__(self, public_keys_dir: str | Path) -> None:
        self.public_keys_dir = Path(public_keys_dir)

    @staticmethod
    def message(operator_id: str, label: str, detection_dict: Dict[str, object]) -> bytes:
        return json.dumps({"operator_id": operator_id, "label": label, "detection_dict": detection_dict}, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def verify(self, operator_id: str, label: str, detection_dict: Dict[str, object], signature_b64: str) -> bool:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except Exception as exc:
            LOGGER.warning("cryptography unavailable; operator signature rejected: %s", exc)
            return False
        key_path = self.public_keys_dir / f"{operator_id}.pub"
        if not key_path.exists():
            LOGGER.warning("operator public key missing: %s", operator_id)
            return False
        try:
            pub = serialization.load_pem_public_key(key_path.read_bytes())
            if not isinstance(pub, Ed25519PublicKey):
                return False
            pub.verify(base64.b64decode(signature_b64), self.message(operator_id, label, detection_dict))
            return True
        except Exception as exc:
            LOGGER.warning("operator signature verification failed for %s: %s", operator_id, exc)
            return False


class HardExampleCollector:
    """Async hard-example collector with quorum and FIFO eviction."""

    def __init__(self, config: HardExampleCollectorConfig, verifier: Optional[SignatureVerifier] = None) -> None:
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.metadata_path = Path(config.metadata_jsonl)
        self.images_dir = self.root_dir / "images"
        self.verifier = verifier or SignatureVerifier(config.public_keys_dir)
        self.queue: queue.Queue[Tuple[np.ndarray, Detection, List[OperatorSignature]]] = queue.Queue(maxsize=config.queue_size)
        self._stop = Event()
        self._last_health = time.time()
        self._degraded_reason: Optional[str] = None
        self._security_events: List[str] = []
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._worker = Thread(target=self._run, name="hard-example-collector", daemon=True)
        if self.config.enabled:
            self._worker.start()

    def maybe_save(self, frame_bgr: np.ndarray, detection: Detection, operator_signatures: Iterable[OperatorSignature | Dict[str, str]] = ()) -> bool:
        if not self.config.enabled:
            return False
        parsed = [sig if isinstance(sig, OperatorSignature) else OperatorSignature(**sig) for sig in operator_signatures]
        uncertain = self.config.uncertain_min_confidence < detection.confidence < self.config.uncertain_max_confidence
        has_feedback = len(parsed) > 0
        if not uncertain and not has_feedback:
            return False
        if not self._storage_available():
            self._degraded_reason = "hard example storage full; blocking new saves"
            return False
        try:
            self.queue.put_nowait((frame_bgr.copy(), detection, parsed))
            self._last_health = time.time()
            return True
        except queue.Full:
            self._degraded_reason = "hard example queue full; dropping save request"
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, detection, signatures = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._save_now(frame, detection, signatures)
                self._evict_fifo()
                if self._degraded_reason and "storage" not in self._degraded_reason:
                    self._degraded_reason = None
            except Exception as exc:
                self._degraded_reason = f"hard example save failed: {exc}"
                LOGGER.exception("Hard example save failed")

    def _save_now(self, frame_bgr: np.ndarray, detection: Detection, signatures: List[OperatorSignature]) -> HardExample:
        detection_dict = detection.serializable_dict()
        valid_by_operator: Dict[str, OperatorSignature] = {}
        labels: List[str] = []
        valid_sigs: List[str] = []
        for sig in signatures:
            if self.verifier.verify(sig.operator_id, sig.label, detection_dict, sig.signature):
                valid_by_operator[sig.operator_id] = sig
            else:
                event = f"operator signature rejected: {sig.operator_id}"
                self._security_events.append(event)
                LOGGER.warning(event)
        for sig in valid_by_operator.values():
            labels.append(sig.label)
            valid_sigs.append(sig.signature)
        ground_truth = len(valid_by_operator) >= self.config.quorum_size
        if signatures and not ground_truth:
            LOGGER.warning("operator feedback quorum not met; example saved as non-ground-truth")

        timestamp = time.time()
        name = f"hard_{int(timestamp * 1000)}.jpg"
        image_path = self.images_dir / name
        x, y, w, h = detection.bbox
        h_img, w_img = frame_bgr.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w_img, int(x + w)), min(h_img, int(y + h))
        crop = frame_bgr[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else frame_bgr
        cv2.imwrite(str(image_path), crop)
        example = HardExample(image_path=str(image_path), detection_dict=detection_dict, operator_labels=labels, signatures=valid_sigs, timestamp=timestamp, ground_truth=ground_truth)
        with self.metadata_path.open("a", encoding="utf-8") as fh:
            payload = example.model_dump() if hasattr(example, "model_dump") else example.dict()
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
        return example

    def _storage_available(self) -> bool:
        usage = shutil.disk_usage(self.root_dir if self.root_dir.exists() else ".")
        return usage.free >= self.config.min_free_bytes

    def _evict_fifo(self) -> None:
        images = sorted(self.images_dir.glob("hard_*.jpg"), key=lambda p: p.stat().st_mtime)
        excess = len(images) - self.config.max_examples
        if excess <= 0:
            return
        for image in images[:excess]:
            try:
                image.unlink()
            except FileNotFoundError:
                pass

    def security_events(self) -> List[str]:
        return list(self._security_events)

    def stop(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def health(self) -> HealthStatus:
        return HealthStatus(module_name="hard_example_collector", status="degraded" if self._degraded_reason else "ok", last_heartbeat=self._last_health, degraded_reason=self._degraded_reason)
