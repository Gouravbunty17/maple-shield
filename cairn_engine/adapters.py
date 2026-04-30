"""
Adapters from existing Maple Shield scripts into CAIRN normalized detections.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .schemas import CairnDetection


def from_mvp_detection(det: Dict[str, Any], frame_w: int, frame_h: int) -> CairnDetection:
    """Convert one maple_shield_mvp.py detection dictionary into CAIRN format."""
    prev_center = det.get("prev_center")
    velocity_px = 0.0
    vx = 0.0
    vy = 0.0
    if prev_center is not None:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        vx = cx - float(prev_center[0])
        vy = cy - float(prev_center[1])
        velocity_px = (vx * vx + vy * vy) ** 0.5

    return CairnDetection(
        track_id=int(det.get("track_id", -1)),
        label=str(det.get("label", "unknown")),
        confidence=float(det.get("conf", 0.0)),
        box=[int(x) for x in det.get("box", [0, 0, 0, 0])],
        frame_w=int(frame_w),
        frame_h=int(frame_h),
        track_confirmed=bool(det.get("track_confirmed", False)),
        velocity_px=float(velocity_px),
        vx=float(vx),
        vy=float(vy),
        persistence_frames=int(det.get("persistence_count", 0)),
        source="maple_shield_mvp",
        raw=det,
    )


def from_motion_risk_detection(det: Dict[str, Any], frame_w: int, frame_h: int) -> CairnDetection:
    """Convert one maple_shield_motion_risk.py detection dictionary into CAIRN format."""
    velocity = det.get("velocity", {}) or {}
    return CairnDetection(
        track_id=int(det.get("track_id", -1)),
        label=str(det.get("label", "unknown")),
        confidence=float(det.get("conf", 0.0)),
        box=[int(x) for x in det.get("box", [0, 0, 0, 0])],
        frame_w=int(frame_w),
        frame_h=int(frame_h),
        track_confirmed=bool(det.get("track_confirmed", False)),
        velocity_px=float(velocity.get("speed", 0.0)),
        vx=float(velocity.get("vx", 0.0)),
        vy=float(velocity.get("vy", 0.0)),
        persistence_frames=int(det.get("persistence_count", 0)),
        source="maple_shield_motion_risk",
        raw=det,
    )


def batch_from_mvp(detections: Iterable[Dict[str, Any]], frame_w: int, frame_h: int) -> List[CairnDetection]:
    return [from_mvp_detection(det, frame_w, frame_h) for det in detections]


def batch_from_motion_risk(detections: Iterable[Dict[str, Any]], frame_w: int, frame_h: int) -> List[CairnDetection]:
    return [from_motion_risk_detection(det, frame_w, frame_h) for det in detections]
