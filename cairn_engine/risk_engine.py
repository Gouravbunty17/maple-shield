"""
CAIRN Risk Engine v2.

This is the decision layer behind Maple Shield. It turns raw detections and
track signals into explainable risk outputs suitable for audit logs, operator
UI, MQTT, CoT, and future sensor-fusion adapters.
"""

from __future__ import annotations

from typing import Iterable, List
import math

from .config import CairnRiskConfig
from .schemas import CairnDetection, CairnRiskResult, CairnThreatLevel, clamp


class CairnRiskEngine:
    def __init__(self, config: CairnRiskConfig | None = None):
        self.config = config or CairnRiskConfig()
        self._drone_labels = {x.lower() for x in self.config.drone_labels}
        self._bird_labels = {x.lower() for x in self.config.bird_labels}
        self._ignore_labels = {x.lower() for x in self.config.ignore_labels}

    def classify_object(self, label: str) -> str:
        normalized = label.strip().lower()
        if normalized in self._drone_labels:
            return "drone"
        if normalized in self._bird_labels:
            return "bird"
        if normalized in self._ignore_labels:
            return "ignore"
        return "unknown"

    def score_many(self, detections: Iterable[CairnDetection]) -> List[CairnRiskResult]:
        return [self.score(det) for det in detections]

    def score(self, det: CairnDetection) -> CairnRiskResult:
        cfg = self.config
        object_type = self.classify_object(det.label)

        factors = {
            "object_type": self._object_factor(object_type),
            "confidence": clamp(det.confidence),
            "track": 1.0 if det.track_confirmed else 0.0,
            "zone": 1.0 if self._in_protected_zone(det) else 0.0,
            "distance": self._distance_factor(det),
            "velocity": self._velocity_factor(det.velocity_px),
            "persistence": self._persistence_factor(det.persistence_frames),
        }

        if object_type == "ignore":
            risk_score = 0.0
            threat = CairnThreatLevel.CLEAR
            reasons = ["ignored class"]
        else:
            risk_score = clamp(
                cfg.weight_object_type * factors["object_type"]
                + cfg.weight_confidence * factors["confidence"]
                + cfg.weight_track * factors["track"]
                + cfg.weight_zone * factors["zone"]
                + cfg.weight_distance * factors["distance"]
                + cfg.weight_velocity * factors["velocity"]
                + cfg.weight_persistence * factors["persistence"]
            )
            threat = self._level_from_score(risk_score)
            reasons = self._reasons(det, object_type, factors)

        return CairnRiskResult(
            track_id=det.track_id,
            label=det.label,
            object_type=object_type,
            threat_level=threat,
            risk_score=round(risk_score, 4),
            confidence=round(float(det.confidence), 4),
            box=det.box,
            in_protected_zone=bool(factors["zone"] >= 1.0),
            factors={k: round(v, 4) for k, v in factors.items()},
            reasons=reasons,
            recommended_operator_action=self._operator_action(threat),
        )

    def _object_factor(self, object_type: str) -> float:
        if object_type == "drone":
            return 1.0
        if object_type == "unknown":
            return 0.45
        if object_type == "bird":
            return 0.20
        return 0.0

    def _in_protected_zone(self, det: CairnDetection) -> bool:
        cx, cy = det.center
        frame_cx = det.frame_w / 2.0
        frame_cy = det.frame_h / 2.0
        radius = self.config.protected_zone_radius_frac * min(det.frame_w, det.frame_h)
        return math.hypot(cx - frame_cx, cy - frame_cy) <= radius

    def _distance_factor(self, det: CairnDetection) -> float:
        if det.area_ratio >= self.config.close_area_ratio:
            return 1.0
        if det.area_ratio >= self.config.medium_area_ratio:
            return 0.55
        return 0.15

    def _velocity_factor(self, velocity_px: float) -> float:
        v = max(0.0, float(velocity_px))
        if v >= self.config.velocity_high_px:
            return 1.0
        if v >= self.config.velocity_medium_px:
            return 0.55
        if v > 0:
            return 0.20
        return 0.0

    def _persistence_factor(self, frames: int) -> float:
        f = max(0, int(frames))
        if f >= self.config.persistence_high_frames:
            return 1.0
        if f >= self.config.persistence_medium_frames:
            return 0.55
        if f > 0:
            return 0.20
        return 0.0

    def _level_from_score(self, score: float) -> CairnThreatLevel:
        if score >= self.config.critical_threshold:
            return CairnThreatLevel.CRITICAL
        if score >= self.config.high_threshold:
            return CairnThreatLevel.HIGH
        if score >= self.config.medium_threshold:
            return CairnThreatLevel.MEDIUM
        if score >= self.config.low_threshold:
            return CairnThreatLevel.LOW
        return CairnThreatLevel.CLEAR

    def _reasons(self, det: CairnDetection, object_type: str, factors: dict[str, float]) -> list[str]:
        reasons: list[str] = []
        if object_type == "drone":
            reasons.append("aerial threat class")
        elif object_type == "unknown":
            reasons.append("unknown aerial object")
        elif object_type == "bird":
            reasons.append("bird-like signature; monitor for false positive")

        if det.confidence >= 0.70:
            reasons.append("high model confidence")
        elif det.confidence >= 0.45:
            reasons.append("medium model confidence")

        if det.track_confirmed:
            reasons.append("confirmed persistent track")
        else:
            reasons.append("unconfirmed track")

        if factors.get("zone", 0) >= 1:
            reasons.append("inside protected zone")
        if factors.get("distance", 0) >= 1:
            reasons.append("close-range size proxy")
        if factors.get("velocity", 0) >= 0.55:
            reasons.append("meaningful motion rate")
        if factors.get("persistence", 0) >= 0.55:
            reasons.append("sustained presence")

        return reasons or ["no elevated risk factors"]

    def _operator_action(self, threat: CairnThreatLevel) -> str:
        if threat == CairnThreatLevel.CRITICAL:
            return "Immediate operator review; cue C-UAS response chain if confirmed."
        if threat == CairnThreatLevel.HIGH:
            return "Prioritize track; verify class and trajectory; prepare response option."
        if threat == CairnThreatLevel.MEDIUM:
            return "Monitor closely; maintain track and request secondary confirmation."
        if threat == CairnThreatLevel.LOW:
            return "Log and monitor; no immediate action."
        return "No action; continue passive monitoring."
