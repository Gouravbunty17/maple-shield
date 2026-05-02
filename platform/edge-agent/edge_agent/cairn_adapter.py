"""CAIRN -> platform detector adapter.

The platform detector contract is deliberately small: emit drone detections
with bounding boxes and confidence. This adapter lets the existing CAIRN engine
sit behind that contract while defensively dropping every non-drone class.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from edge_agent.detector import Detection


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)


_ensure_repo_root_on_path()

from cairn_engine import CairnDetection, CairnEngine  # noqa: E402
from cairn_engine import __version__ as CAIRN_PACKAGE_VERSION  # noqa: E402


EXPECTED_CAIRN_VERSION = "2.0.0-dev"
CairnDetectionProvider = Callable[[int, int, int], Iterable[CairnDetection]]


class CairnVersionWarning(RuntimeWarning):
    """Raised when the adapter sees an untested CAIRN minor version."""


def _major_minor(version: str) -> Tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def check_cairn_version(actual: str = CAIRN_PACKAGE_VERSION) -> bool:
    """Return compatibility and warn if CAIRN minor version drifted."""
    actual_mm = _major_minor(actual)
    expected_mm = _major_minor(EXPECTED_CAIRN_VERSION)
    compatible = actual_mm is not None and actual_mm == expected_mm
    if not compatible:
        warnings.warn(
            f"CAIRN adapter tested against {EXPECTED_CAIRN_VERSION}; found {actual}.",
            CairnVersionWarning,
            stacklevel=2,
        )
    return compatible


class CairnSourceDetector:
    """Detector Protocol implementation backed by ``CairnEngine``.

    ``detection_provider`` is intentionally injected. The adapter owns schema
    translation and class restriction; camera/model-specific detection remains
    outside this file so tests do not need model weights.
    """

    def __init__(
        self,
        engine: Optional[CairnEngine] = None,
        detection_provider: Optional[CairnDetectionProvider] = None,
        session_id: str = "platform-cairn",
        fps: float = 0.0,
    ):
        check_cairn_version()
        self.engine = engine or CairnEngine()
        self.detection_provider = detection_provider or self._empty_provider
        self.session_id = session_id
        self.fps = fps
        self.last_record = None

    @staticmethod
    def _empty_provider(frame_idx: int, frame_w: int, frame_h: int) -> Iterable[CairnDetection]:
        return []

    def detect(self, frame_idx: int, frame_w: int, frame_h: int) -> List[Detection]:
        cairn_detections = list(self.detection_provider(frame_idx, frame_w, frame_h))
        record = self.engine.process_frame(
            frame_id=frame_idx,
            detections=cairn_detections,
            session_id=self.session_id,
            frame_w=frame_w,
            frame_h=frame_h,
            fps=self.fps,
        )
        self.last_record = record
        return [det for det in (self._to_platform_detection(risk) for risk in record.risks) if det]

    def detect_frame(self, frame_idx: int, frame) -> List[Detection]:
        """Convenience wrapper for sources that pass the actual frame object."""
        frame_h, frame_w = frame.shape[:2]
        return self.detect(frame_idx, frame_w, frame_h)

    def _to_platform_detection(self, risk) -> Optional[Detection]:
        label = str(getattr(risk, "label", "")).strip().lower()
        object_type = self._classify(label, fallback=str(getattr(risk, "object_type", "")))
        if object_type != "drone":
            return None

        bbox = tuple(float(x) for x in getattr(risk, "box"))
        if len(bbox) != 4:
            return None

        threat_level = getattr(risk, "threat_level", None)
        threat_label = threat_level.label() if hasattr(threat_level, "label") else str(threat_level)
        track_id = f"cairn-{getattr(risk, 'track_id')}"

        return Detection(
            cls="drone",
            confidence=float(getattr(risk, "confidence")),
            bbox=bbox,  # type: ignore[arg-type]
            track_id=track_id,
            raw={
                "cairn_track_id": getattr(risk, "track_id"),
                "cairn_object_type": object_type,
                "cairn_threat_level": threat_label,
                "cairn_risk_score": float(getattr(risk, "risk_score", 0.0)),
                "cairn_reasons": list(getattr(risk, "reasons", [])),
            },
        )

    def _classify(self, label: str, fallback: str) -> str:
        classifier = getattr(getattr(self.engine, "risk_engine", None), "classify_object", None)
        if callable(classifier):
            return str(classifier(label)).strip().lower()
        return fallback.strip().lower()
