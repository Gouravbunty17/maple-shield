"""CPU-friendly track kinematics for Cairn-Edge."""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Iterable, List, Optional, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class KinematicFingerprint:
    track_id: int
    speed_px_s: float
    heading_deg: float
    acceleration_px_s2: float
    jerk_px_s3: float
    hover_score: float
    bird_like_score: float
    three_second_prediction: Point


def _velocity(p0: Point, p1: Point, dt: float) -> Point:
    if dt <= 0:
        return (0.0, 0.0)
    return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)


def compute_kinematic_fingerprint(track_id: int, centers: List[Point], timestamps: List[float]) -> Optional[KinematicFingerprint]:
    """Compute lightweight motion descriptors from recent center history.

    The caller should pass the most recent points for one track. This does not
    perform identity tracking; it only summarizes track motion for risk and
    bird/drone disambiguation.
    """
    if len(centers) < 2 or len(centers) != len(timestamps):
        return None

    p_prev = centers[-2]
    p_now = centers[-1]
    dt = max(1e-3, timestamps[-1] - timestamps[-2])
    vx, vy = _velocity(p_prev, p_now, dt)
    speed = hypot(vx, vy)
    heading = (degrees(atan2(vy, vx)) + 360.0) % 360.0

    acceleration = 0.0
    jerk = 0.0
    if len(centers) >= 3:
        dt_prev = max(1e-3, timestamps[-2] - timestamps[-3])
        pvx, pvy = _velocity(centers[-3], centers[-2], dt_prev)
        ax = (vx - pvx) / dt
        ay = (vy - pvy) / dt
        acceleration = hypot(ax, ay)

    if len(centers) >= 4:
        # Approximate jerk from speed deltas. Good enough for a CPU-side cue.
        old_vx, old_vy = _velocity(centers[-4], centers[-3], max(1e-3, timestamps[-3] - timestamps[-4]))
        old_speed = hypot(old_vx, old_vy)
        prev_speed = hypot(*_velocity(centers[-3], centers[-2], max(1e-3, timestamps[-2] - timestamps[-3])))
        jerk = abs((speed - prev_speed) - (prev_speed - old_speed)) / dt

    hover_score = max(0.0, min(1.0, 1.0 - (speed / 18.0)))
    bird_like_score = max(0.0, min(1.0, (jerk / 60.0) + (acceleration / 120.0)))
    prediction = (p_now[0] + vx * 3.0, p_now[1] + vy * 3.0)

    return KinematicFingerprint(
        track_id=int(track_id),
        speed_px_s=float(speed),
        heading_deg=float(heading),
        acceleration_px_s2=float(acceleration),
        jerk_px_s3=float(jerk),
        hover_score=float(hover_score),
        bird_like_score=float(bird_like_score),
        three_second_prediction=prediction,
    )


def summarize_swarm_candidates(points: Iterable[Point], eps_px: float = 80.0, min_tracks: int = 3) -> bool:
    """Tiny dependency-free clustering cue.

    This is not a replacement for DBSCAN, but it provides a safe fallback on
    minimal edge images where sklearn is not installed.
    """
    pts = list(points)
    if len(pts) < min_tracks:
        return False
    for i, p in enumerate(pts):
        neighbors = 1
        for j, q in enumerate(pts):
            if i == j:
                continue
            if hypot(p[0] - q[0], p[1] - q[1]) <= eps_px:
                neighbors += 1
        if neighbors >= min_tracks:
            return True
    return False
