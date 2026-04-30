"""
CAIRN Engine configuration.

This keeps operational thresholds outside the detector loop so the same engine
can run demo, outdoor test, Arctic, or critical-infrastructure profiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class CairnRiskConfig:
    # Protected zone geometry
    protected_zone_radius_frac: float = 0.25
    forward_sector_width_frac: float = 0.60

    # Distance proxy from bounding-box size
    close_area_ratio: float = 0.15
    medium_area_ratio: float = 0.05

    # Scoring weights. Keep total <= 1.0 before escalation.
    weight_object_type: float = 0.25
    weight_confidence: float = 0.15
    weight_track: float = 0.10
    weight_zone: float = 0.15
    weight_distance: float = 0.15
    weight_velocity: float = 0.10
    weight_persistence: float = 0.10

    # Thresholds
    low_threshold: float = 0.15
    medium_threshold: float = 0.40
    high_threshold: float = 0.65
    critical_threshold: float = 0.85

    # Motion thresholds, in pixels/frame until calibrated with camera geometry
    velocity_medium_px: float = 4.0
    velocity_high_px: float = 10.0
    persistence_medium_frames: int = 10
    persistence_high_frames: int = 30

    # Operational taxonomy
    drone_labels: List[str] = field(default_factory=lambda: [
        "drone", "uav", "quadcopter", "airplane", "helicopter", "kite"
    ])
    bird_labels: List[str] = field(default_factory=lambda: ["bird"])
    ignore_labels: List[str] = field(default_factory=lambda: [
        "person", "car", "truck", "bus", "bicycle", "motorcycle", "traffic light", "stop sign"
    ])


@dataclass
class CairnEngineConfig:
    engine_name: str = "CAIRN"
    engine_version: str = "2.0.0-dev"
    session_prefix: str = "cairn"
    risk: CairnRiskConfig = field(default_factory=CairnRiskConfig)

    # Output controls
    min_c2_threat_level: str = "MEDIUM"
    enable_audit_log: bool = True
    enable_health_record: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "CairnEngineConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        risk_payload = payload.pop("risk", {})
        return cls(risk=CairnRiskConfig(**risk_payload), **payload)

    def to_dict(self) -> Dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
