"""Geofence and dynamic risk scoring for Cairn-Edge."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .models import HealthStatus, RiskAssessment, Track

Action = Literal["ignore", "monitor", "alert", "escalate"]


class GeofenceZone(BaseModel):
    name: str
    polygon: List[Tuple[float, float]] = Field(min_length=3)  # [lat, lon]
    altitude_min_m: float = -1000.0
    altitude_max_m: float = 100000.0
    action: Action = "monitor"
    risk_multiplier: float = Field(default=1.0, ge=0.0)


class GeofenceConfig(BaseModel):
    zones: List[GeofenceZone] = Field(default_factory=list)


@dataclass(frozen=True)
class ZoneMatch:
    zone: GeofenceZone
    adjusted_score: float


def point_in_polygon(lat: float, lon: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test over lat/lon pairs."""
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi, xi = polygon[i]
        yj, xj = polygon[j]
        intersects = ((xi > lon) != (xj > lon)) and (
            lat < (yj - yi) * (lon - xi) / ((xj - xi) or 1e-12) + yi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


class GeofenceEngine:
    """Evaluate tracks against altitude-aware geofence zones."""

    def __init__(self, config_path: str | Path, cot_emitter: Optional[object] = None) -> None:
        self.config_path = Path(config_path)
        self.cot_emitter = cot_emitter
        self.zones: List[GeofenceZone] = []
        self.enabled = False
        self._last_health = time.time()
        self._degraded_reason: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        try:
            raw = self._load_yaml_or_json(self.config_path)
            cfg = GeofenceConfig(**raw)
            self.zones = cfg.zones
            self.enabled = True
            self._degraded_reason = None
        except Exception as exc:
            self.zones = []
            self.enabled = False
            self._degraded_reason = f"geofence config invalid: {exc}"
        self._last_health = time.time()

    @staticmethod
    def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(str(path))
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML required for YAML geofence files: pip install pyyaml") from exc
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError("geofence config must be a mapping")
        return data

    def _active_matches(self, track: Track) -> List[ZoneMatch]:
        matches: List[ZoneMatch] = []
        if not self.enabled:
            return matches
        for zone in self.zones:
            if not (zone.altitude_min_m <= track.alt <= zone.altitude_max_m):
                continue
            if point_in_polygon(track.lat, track.lon, zone.polygon):
                score = min(100.0, max(0.0, track.kinematic_risk * zone.risk_multiplier))
                matches.append(ZoneMatch(zone=zone, adjusted_score=score))
        return matches

    def evaluate_track(self, track: Track) -> RiskAssessment:
        if not self.enabled:
            return RiskAssessment(score=track.kinematic_risk, action="monitor", zone_name=None, reason="geofence disabled")
        matches = self._active_matches(track)
        if not matches:
            return RiskAssessment(score=track.kinematic_risk, action="monitor", zone_name=None, reason="no active geofence zone")
        best = max(matches, key=lambda m: (m.zone.risk_multiplier, m.adjusted_score))
        reason = (
            f"track inside zone '{best.zone.name}', base_risk={track.kinematic_risk:.1f}, "
            f"multiplier={best.zone.risk_multiplier:.2f}"
        )
        assessment = RiskAssessment(score=best.adjusted_score, action=best.zone.action, zone_name=best.zone.name, reason=reason)
        if assessment.action in {"alert", "escalate"}:
            self.emit_cot(track, assessment)
        return assessment

    def check_crossing(self, prev_track: Track, curr_track: Track) -> Optional[Tuple[str, str]]:
        if not self.enabled:
            return None
        best_event: Optional[Tuple[str, str, float]] = None
        for zone in self.zones:
            prev_inside = (
                zone.altitude_min_m <= prev_track.alt <= zone.altitude_max_m
                and point_in_polygon(prev_track.lat, prev_track.lon, zone.polygon)
            )
            curr_inside = (
                zone.altitude_min_m <= curr_track.alt <= zone.altitude_max_m
                and point_in_polygon(curr_track.lat, curr_track.lon, zone.polygon)
            )
            if prev_inside == curr_inside:
                continue
            direction = "entering" if curr_inside else "exiting"
            if best_event is None or zone.risk_multiplier > best_event[2]:
                best_event = (zone.name, direction, zone.risk_multiplier)
        if best_event is None:
            return None
        return best_event[0], best_event[1]

    def emit_cot(self, track: Track, assessment: RiskAssessment) -> None:
        if self.cot_emitter is None:
            return
        type_code = "a-h-A-M-F-Q" if assessment.action == "escalate" else "a-u-A-M-F-Q"
        remarks = f"CAIRN geofence {assessment.action}: {assessment.zone_name}; score={assessment.score:.1f}; {assessment.reason}"
        try:
            if hasattr(self.cot_emitter, "emit"):
                self.cot_emitter.emit(
                    type_code=type_code,
                    lat=track.lat,
                    lon=track.lon,
                    remarks=remarks,
                    callsign=f"track-{track.track_id}",
                )
                return
            if hasattr(self.cot_emitter, "send_xml"):
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30))
                xml = (
                    f'<event version="2.0" uid="track-{track.track_id}" type="{type_code}" how="m-g" '
                    f'time="{now}" start="{now}" stale="{stale}">'
                    f'<point lat="{track.lat:.7f}" lon="{track.lon:.7f}" hae="{track.alt:.1f}" ce="9999999" le="9999999" />'
                    f'<detail><remarks>{remarks}</remarks><cairn reporting_only="true" threat_action="{assessment.action}" /></detail></event>'
                )
                self.cot_emitter.send_xml(xml)
        except Exception as exc:
            self._degraded_reason = f"geofence CoT emission failed: {exc}"

    def health(self) -> HealthStatus:
        return HealthStatus(
            module_name="geofence_engine",
            status="ok" if self.enabled and not self._degraded_reason else "degraded",
            last_heartbeat=self._last_health,
            degraded_reason=self._degraded_reason,
        )
