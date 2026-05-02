"""Pydantic models / data contracts for Maple Shield command-api.

These models are the contract for both the HTTP boundary and the
fusion-engine -> command-api boundary. They are intentionally explicit and
narrow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------- enums ----------


class Severity(str, Enum):
    info = "info"
    low = "low"
    med = "med"
    high = "high"


class IncidentStatus(str, Enum):
    new = "new"
    acknowledged = "acknowledged"
    reviewed = "reviewed"
    closed = "closed"


# ---------- detection / track / alert ----------


class Detection(BaseModel):
    """A single per-frame detection. Only the 'drone' class is supported."""
    cls: str = Field(..., pattern="^drone$")  # restricted by design
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)


class DetectionFrame(BaseModel):
    """The packet edge-agent sends to fusion-engine."""
    frame_id: str = Field(default_factory=lambda: str(uuid4()))
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    camera_id: str
    image_size: Tuple[int, int]  # w, h
    detections: List[Detection] = []


class Track(BaseModel):
    """A track maintained by fusion-engine."""
    track_id: str
    camera_id: str
    first_seen: datetime
    last_seen: datetime
    n_observations: int
    smoothed_bbox: Tuple[float, float, float, float]
    velocity_px_s: Tuple[float, float]
    max_confidence: float


class Alert(BaseModel):
    """A scored alert raised by fusion-engine and persisted by command-api."""
    alert_id: str = Field(default_factory=lambda: f"alt-{uuid4().hex[:12]}")
    track_id: str
    camera_id: str
    severity: Severity
    rule: str
    score: float = Field(..., ge=0.0, le=1.0)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_b64: Optional[str] = None  # tiny JPEG, optional


# ---------- incidents ----------


class IncidentNote(BaseModel):
    operator_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    text: str


class Incident(BaseModel):
    incident_id: str = Field(default_factory=lambda: f"inc-{uuid4().hex[:10]}")
    status: IncidentStatus = IncidentStatus.new
    alert_ids: List[str] = []
    notes: List[IncidentNote] = []
    created_ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------- audit ----------


class AuditEntry(BaseModel):
    seq: int
    ts: datetime
    operator_id: str
    action: str
    target: Optional[str] = None
    payload: dict = {}
    prev_hash: str
    hash: str
