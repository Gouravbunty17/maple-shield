"""
CAIRN Engine core schemas.

CAIRN is the internal detection engine behind Maple Shield. These schemas keep
risk scoring, audit logging, replay, C2 adapters, and future sensor-fusion code
working from one stable contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional
import time


class CairnThreatLevel(IntEnum):
    CLEAR = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def label(self) -> str:
        return self.name


@dataclass
class CairnDetection:
    """Normalized object detection + track signal for CAIRN."""

    track_id: int
    label: str
    confidence: float
    box: List[int]
    frame_w: int
    frame_h: int
    track_confirmed: bool = False
    velocity_px: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    persistence_frames: int = 0
    source: str = "eo_camera"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area_ratio(self) -> float:
        x1, y1, x2, y2 = self.box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        frame_area = max(1, self.frame_w * self.frame_h)
        return area / frame_area


@dataclass
class CairnRiskResult:
    """CAIRN output for one tracked detection."""

    track_id: int
    label: str
    object_type: str
    threat_level: CairnThreatLevel
    risk_score: float
    confidence: float
    box: List[int]
    in_protected_zone: bool
    factors: Dict[str, float]
    reasons: List[str]
    recommended_operator_action: str

    def to_json(self) -> Dict[str, Any]:
        data = asdict(self)
        data["threat_level"] = self.threat_level.label()
        return data


@dataclass
class CairnFrameRecord:
    """Frame-level CAIRN output suitable for JSONL logging and C2 adapters."""

    frame: int
    ts: float
    session_id: str
    frame_w: int
    frame_h: int
    fps: float
    infer_ms: Optional[float]
    max_threat_level: CairnThreatLevel
    max_risk_score: float
    risks: List[CairnRiskResult]
    health: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema": "cairn.frame.v1",
            "frame": self.frame,
            "ts": self.ts,
            "iso_time": iso_time(self.ts),
            "session_id": self.session_id,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "fps": round(float(self.fps), 3),
            "infer_ms": None if self.infer_ms is None else round(float(self.infer_ms), 3),
            "max_threat_level": self.max_threat_level.label(),
            "max_risk_score": round(float(self.max_risk_score), 3),
            "risks": [risk.to_json() for risk in self.risks],
            "health": self.health,
        }


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def iso_time(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
