"""Track manager.

Pure-Python, NumPy-only constant-velocity tracker. Each track has a smoothed
bbox and a velocity. New detections are associated greedily by IoU within a
configurable threshold. Tracks that go un-observed for `max_misses` updates
are retired.

Deliberately simple: this is a passive monitoring component. There are no
prediction-into-target operations, no engagement, no targeting — only an
estimator over observed locations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from uuid import uuid4

import numpy as np


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    track_id: str
    camera_id: str
    bbox: Tuple[float, float, float, float]
    velocity: Tuple[float, float] = (0.0, 0.0)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    n_obs: int = 1
    max_conf: float = 0.0
    last_conf: float = 0.0
    misses: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class Tracker:
    def __init__(self, iou_thresh: float = 0.2, max_misses: int = 5,
                 smoothing: float = 0.4):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.smoothing = smoothing  # exponential smoothing weight on new bbox
        self.tracks: dict[str, Track] = {}

    def update(self, camera_id: str, ts: float,
               detections: List[dict]) -> List[Track]:
        """Associate detections to existing tracks, return all live tracks.

        `detections` is a list of {bbox, confidence, cls}. Only `drone` class
        is accepted; anything else is ignored as a defensive measure.
        """
        unmatched_dets = list(range(len(detections)))
        for det in detections:
            if det.get("cls") != "drone":
                # defensive; the model contract restricts class but we
                # double-check at runtime.
                continue

        # Greedy IoU matching: for each track, take the best unmatched detection
        for tid, trk in list(self.tracks.items()):
            if trk.camera_id != camera_id:
                continue
            best_idx, best_iou = -1, 0.0
            for di in list(unmatched_dets):
                d = detections[di]
                if d.get("cls") != "drone":
                    continue
                cand = iou(trk.bbox, tuple(d["bbox"]))
                if cand > best_iou:
                    best_iou = cand
                    best_idx = di
            if best_iou >= self.iou_thresh and best_idx >= 0:
                d = detections[best_idx]
                old_cx, old_cy = trk.center
                # exponentially smooth the bbox
                a = self.smoothing
                nb = tuple(a * d["bbox"][i] + (1 - a) * trk.bbox[i] for i in range(4))
                trk.bbox = nb  # type: ignore[assignment]
                new_cx, new_cy = trk.center
                dt = max(1e-3, ts - trk.last_seen)
                trk.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)
                trk.last_seen = ts
                trk.last_conf = float(d["confidence"])
                trk.max_conf = max(trk.max_conf, float(d["confidence"]))
                trk.n_obs += 1
                trk.misses = 0
                unmatched_dets.remove(best_idx)
            else:
                trk.misses += 1

        # any remaining unmatched detections become new tracks
        for di in unmatched_dets:
            d = detections[di]
            if d.get("cls") != "drone":
                continue
            tid = f"trk-{uuid4().hex[:8]}"
            self.tracks[tid] = Track(
                track_id=tid, camera_id=camera_id,
                bbox=tuple(d["bbox"]),
                first_seen=ts, last_seen=ts,
                n_obs=1,
                max_conf=float(d["confidence"]),
                last_conf=float(d["confidence"]),
            )

        # retire stale tracks
        for tid, trk in list(self.tracks.items()):
            if trk.misses > self.max_misses:
                del self.tracks[tid]

        return list(self.tracks.values())
