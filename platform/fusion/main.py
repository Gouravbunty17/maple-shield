"""fusion-engine HTTP shim.

Receives DetectionFrame from edge-agent on POST /detections, runs the
tracker + scorer, and posts any alerts to command-api.
"""

from __future__ import annotations

import os
from datetime import timezone
from typing import Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from fusion.scorer import ScorerConfig, score_track
from fusion.tracker import Tracker

COMMAND_API = os.environ.get("MAPLE_SHIELD_COMMAND_API", "http://localhost:8080")

app = FastAPI(
    title="Maple Shield - fusion-engine",
    version="0.1.0",
    description="Track state + alert scoring. Decision support only - never originates an action against a target.",
)

_tracker = Tracker()
_cfg = ScorerConfig()
# In-memory dedup so we don't post the same alert twice per track.
_last_alert_rule_for_track: dict[str, str] = {}


class DetectionIn(BaseModel):
    cls: str
    confidence: float
    bbox: list[float]
    track_id: Optional[str] = None


class FrameIn(BaseModel):
    frame_id: Optional[str] = None
    ts: Optional[float] = None  # unix seconds; if None, server time
    camera_id: str
    image_size: list[int]
    detections: list[DetectionIn] = []


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "fusion-engine"}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}


@app.get("/tracks")
def tracks():
    return [
        {
            "track_id": t.track_id, "camera_id": t.camera_id,
            "bbox": list(t.bbox), "velocity": list(t.velocity),
            "n_obs": t.n_obs, "max_conf": t.max_conf,
            "last_conf": t.last_conf,
            "first_seen": t.first_seen, "last_seen": t.last_seen,
        }
        for t in _tracker.tracks.values()
    ]


@app.post("/detections")
def post_detections(frame: FrameIn):
    import time as _time
    ts = frame.ts if frame.ts is not None else _time.time()
    det_list = [
        {"cls": d.cls, "confidence": d.confidence, "bbox": d.bbox, "track_id": d.track_id}
        for d in frame.detections
    ]
    live_tracks = _tracker.update(frame.camera_id, ts, det_list)

    # GC: drop dedup entries for tracks that are no longer live
    live_ids = {t.track_id for t in live_tracks}
    for tid in list(_last_alert_rule_for_track):
        if tid not in live_ids:
            _last_alert_rule_for_track.pop(tid, None)

    posted_alerts = []
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        for trk in live_tracks:
            sa = score_track(trk, _cfg)
            if sa is None:
                continue
            # only post when the rule changes (debounce)
            if _last_alert_rule_for_track.get(sa.track_id) == sa.rule:
                continue
            _last_alert_rule_for_track[sa.track_id] = sa.rule
            payload = {
                "track_id": sa.track_id, "camera_id": sa.camera_id,
                "severity": sa.severity, "rule": sa.rule, "score": sa.score,
            }
            try:
                client.post(f"{COMMAND_API}/alerts", json=payload)
                posted_alerts.append(payload)
            except httpx.HTTPError as e:
                # In offline / dev mode, just log; we still return what we tried.
                payload["error"] = str(e)
                posted_alerts.append(payload)

    return {
        "frame_id": frame.frame_id,
        "n_tracks": len(live_tracks),
        "posted_alerts": posted_alerts,
    }
