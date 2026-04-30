"""Shared Pydantic models for Cairn-Edge tactical and sensor modules.

These models are intentionally compact. They are safe to serialize over the mesh,
write to JSONL, and reuse in CPU hot paths on Jetson Orin Nano.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class Track(BaseModel):
    """Normalized air/sky track used by swarm, geofence, mesh, and evidence modules."""

    track_id: str
    lat: float
    lon: float
    alt: float = 0.0
    velocity: float = 0.0  # meters/second when calibrated, otherwise best estimate
    heading: float = 0.0  # degrees true/estimated, 0-360
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    class_id: str = "unknown"
    kinematic_risk: float = Field(default=0.0, ge=0.0, le=100.0)
    timestamp: float = Field(default_factory=time.time)

    def as_payload(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()  # pydantic v2
        return self.dict()  # pydantic v1


class ThermalDetection(BaseModel):
    """Thermal hot-spot detection aligned to RGB frame coordinates."""

    bbox: Tuple[int, int, int, int]
    temperature_c: float
    area_px: int = Field(ge=0)


class KLVFrame(BaseModel):
    """MISB/STANAG metadata frame for periodic evidence export."""

    timestamp_ns: int
    mandatory_tags: Dict[int, Any]
    warning_flags: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Geofence/risk decision for one track."""

    score: float = Field(ge=0.0, le=100.0)
    action: Literal["ignore", "monitor", "alert", "escalate"]
    zone_name: Optional[str] = None
    reason: str


class MeshMessage(BaseModel):
    """Authenticated message exchanged between Cairn-Edge nodes."""

    type: Literal["TrackUpdate", "Heartbeat"]
    source_node: str
    sequence_number: int = Field(ge=0)
    timestamp: float
    signature: str = ""
    payload: Dict[str, Any]

    def as_payload(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class HealthStatus(BaseModel):
    """Module health record for central HealthMonitor."""

    module_name: str
    status: Literal["ok", "degraded", "error"]
    last_heartbeat: float
    degraded_reason: Optional[str] = None
