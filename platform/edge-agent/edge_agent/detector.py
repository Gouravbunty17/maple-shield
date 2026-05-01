"""Detector interface and a deterministic mock implementation.

The real CAIRN detection engine would conform to the same `Detector`
protocol. Shipping the mock keeps the demo reproducible and avoids
shipping any model weights.

Important: by design the detector emits ONLY the `drone` class. There is
no person/face/vehicle output and no biometric extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import List, Protocol, Tuple


@dataclass
class Detection:
    cls: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2


class Detector(Protocol):
    def detect(self, frame_idx: int, frame_w: int, frame_h: int) -> List[Detection]:
        ...


class CairnMockDetector:
    """Deterministic mock detector.

    Walks a small bbox along a sinusoidal path through the frame and emits
    a `drone` detection on most frames. Confidence varies in a known pattern
    so alert thresholds are exercised end-to-end.
    """

    def __init__(self, p_detect: float = 0.92, base_conf: float = 0.78):
        self.p_detect = p_detect
        self.base_conf = base_conf

    def detect(self, frame_idx: int, frame_w: int, frame_h: int) -> List[Detection]:
        # deterministic miss pattern: drop every 13th frame
        if frame_idx % 13 == 0:
            return []
        # path
        cx = 60 + (frame_idx * 6) % max(1, (frame_w - 120))
        cy = int(frame_h / 2 + sin(frame_idx / 7.0) * frame_h / 6)
        w, h = 40, 30
        x1 = max(0, cx - w // 2); y1 = max(0, cy - h // 2)
        x2 = min(frame_w, x1 + w); y2 = min(frame_h, y1 + h)
        # confidence ramps up over the first ~30 frames so we exercise the
        # severity ladder (info -> low -> med -> high)
        conf = min(0.97, self.base_conf + 0.005 * frame_idx)
        return [Detection(cls="drone", confidence=conf, bbox=(x1, y1, x2, y2))]
