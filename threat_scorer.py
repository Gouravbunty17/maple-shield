"""
threat_scorer.py — Maple Shield Threat Scoring

Classifies detections into threat categories and assigns a threat level
(CLEAR / LOW / MEDIUM / HIGH / CRITICAL) based on:
  - Object class (drone vs bird vs other)
  - Detection confidence
  - Track confirmation status
  - Trajectory (velocity, approach angle)
  - Proximity to a defined protected zone

Designed to be stateless per call — the caller (tracker pipeline) owns track state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Threat levels
# ---------------------------------------------------------------------------

class ThreatLevel(IntEnum):
    CLEAR    = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def label(self) -> str:
        return self.name

    def color_bgr(self) -> tuple[int, int, int]:
        """OpenCV BGR color for overlay rendering."""
        return {
            ThreatLevel.CLEAR:    (0,   200, 0),
            ThreatLevel.LOW:      (0,   200, 200),
            ThreatLevel.MEDIUM:   (0,   165, 255),
            ThreatLevel.HIGH:     (0,   0,   255),
            ThreatLevel.CRITICAL: (0,   0,   180),
        }[self]


# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------

# COCO classes that map to "aerial drone" signatures
DRONE_CLASSES = {
    # Direct: not in COCO80 by name, but fine-tuned models may output these
    "drone", "uav", "quadcopter",
    # Proxy classes used in transfer-learned detectors
    "kite",       # common false-positive / similar silhouette
    "airplane",   # fixed-wing UAS
    "helicopter", # rotary UAS
}

BIRD_CLASSES = {"bird"}

# Classes that are never threats (filter noise)
IGNORE_CLASSES = {
    "person", "car", "truck", "bus", "bicycle", "motorcycle",
    "traffic light", "stop sign",
}


def classify_object(label: str) -> str:
    """Return 'drone', 'bird', 'ignore', or 'unknown'."""
    label_lower = label.lower()
    if label_lower in DRONE_CLASSES:
        return "drone"
    if label_lower in BIRD_CLASSES:
        return "bird"
    if label_lower in IGNORE_CLASSES:
        return "ignore"
    return "unknown"


# ---------------------------------------------------------------------------
# Scoring config
# ---------------------------------------------------------------------------

@dataclass
class ScorerConfig:
    # Confidence thresholds
    conf_drone_high: float = 0.70      # High-confidence drone detection
    conf_drone_medium: float = 0.45    # Medium-confidence

    # Track confirmation (min hits before escalating)
    require_confirmed_for_high: bool = True

    # Velocity thresholds (pixels/frame, normalized to 640px width)
    velocity_approach_threshold: float = 8.0   # px/frame — fast approach
    velocity_hover_threshold: float = 2.0      # px/frame — hovering

    # Protected zone: fraction of frame (0.0–1.0), centered
    zone_radius_frac: float = 0.25

    # Frame dimensions (updated at runtime)
    frame_w: int = 1280
    frame_h: int = 720


# ---------------------------------------------------------------------------
# Detection scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoredDetection:
    track_id: int
    label: str
    object_type: str          # drone / bird / ignore / unknown
    conf: float
    box: list[int]            # [x1, y1, x2, y2]
    track_confirmed: bool
    threat_level: ThreatLevel
    threat_score: float       # 0.0–1.0 continuous
    in_zone: bool
    velocity_px: float        # magnitude, px/frame
    notes: list[str] = field(default_factory=list)


def _box_center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def score_detection(
    track_id: int,
    label: str,
    conf: float,
    box: list[int],
    track_confirmed: bool,
    prev_center: Optional[tuple[float, float]],
    config: ScorerConfig,
) -> ScoredDetection:
    """
    Score a single tracked detection.

    Args:
        track_id:       Tracker-assigned ID.
        label:          Class label string.
        conf:           Detection confidence [0,1].
        box:            [x1, y1, x2, y2] in frame pixels.
        track_confirmed: True if track has reached min_hits.
        prev_center:    Previous frame center (x, y) for velocity estimate.
        config:         Scorer configuration.

    Returns:
        ScoredDetection with threat_level and threat_score.
    """
    notes = []
    obj_type = classify_object(label)

    # Velocity
    center = _box_center(box)
    velocity_px = 0.0
    if prev_center is not None:
        velocity_px = _distance(center, prev_center)

    # Zone check — is the object inside the protected zone?
    frame_cx = config.frame_w / 2.0
    frame_cy = config.frame_h / 2.0
    zone_r = config.zone_radius_frac * min(config.frame_w, config.frame_h)
    in_zone = _distance(center, (frame_cx, frame_cy)) <= zone_r

    # Base score accumulator
    score = 0.0

    if obj_type == "ignore":
        return ScoredDetection(
            track_id=track_id, label=label, object_type=obj_type,
            conf=conf, box=box, track_confirmed=track_confirmed,
            threat_level=ThreatLevel.CLEAR, threat_score=0.0,
            in_zone=in_zone, velocity_px=velocity_px,
            notes=["ignored class"],
        )

    # --- Scoring rules ---

    # 1. Object type
    if obj_type == "drone":
        score += 0.50
        notes.append("drone class")
    elif obj_type == "bird":
        score += 0.10
        notes.append("bird class")
    elif obj_type == "unknown":
        score += 0.20
        notes.append("unknown aerial object")

    # 2. Confidence
    if conf >= config.conf_drone_high:
        score += 0.20
        notes.append(f"high conf ({conf:.2f})")
    elif conf >= config.conf_drone_medium:
        score += 0.10
        notes.append(f"medium conf ({conf:.2f})")

    # 3. Track confirmation
    if track_confirmed:
        score += 0.10
        notes.append("track confirmed")
    else:
        notes.append("track unconfirmed")

    # 4. Velocity
    if velocity_px >= config.velocity_approach_threshold:
        score += 0.15
        notes.append(f"fast approach ({velocity_px:.1f}px/fr)")
    elif velocity_px <= config.velocity_hover_threshold and track_confirmed:
        score += 0.05
        notes.append(f"hovering ({velocity_px:.1f}px/fr)")

    # 5. Zone proximity
    if in_zone:
        score += 0.15
        notes.append("inside protected zone")

    score = min(score, 1.0)

    # Map score to threat level
    if score >= 0.85:
        level = ThreatLevel.CRITICAL
    elif score >= 0.65:
        level = ThreatLevel.HIGH
    elif score >= 0.40:
        level = ThreatLevel.MEDIUM
    elif score >= 0.15:
        level = ThreatLevel.LOW
    else:
        level = ThreatLevel.CLEAR

    # Downgrade HIGH → MEDIUM if track not confirmed and config requires it
    if level == ThreatLevel.HIGH and config.require_confirmed_for_high and not track_confirmed:
        level = ThreatLevel.MEDIUM
        notes.append("downgraded: track unconfirmed")

    return ScoredDetection(
        track_id=track_id,
        label=label,
        object_type=obj_type,
        conf=conf,
        box=box,
        track_confirmed=track_confirmed,
        threat_level=level,
        threat_score=round(score, 3),
        in_zone=in_zone,
        velocity_px=round(velocity_px, 2),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = ScorerConfig(frame_w=1280, frame_h=720)

    test_cases = [
        dict(track_id=1, label="drone", conf=0.82, box=[600, 340, 680, 380],
             track_confirmed=True, prev_center=(590, 330)),
        dict(track_id=2, label="bird",  conf=0.60, box=[200, 100, 240, 130],
             track_confirmed=True, prev_center=(195, 98)),
        dict(track_id=3, label="kite",  conf=0.48, box=[640, 360, 700, 400],
             track_confirmed=False, prev_center=None),
        dict(track_id=4, label="airplane", conf=0.75, box=[630, 355, 700, 395],
             track_confirmed=True, prev_center=(610, 340)),
    ]

    print("=== Threat Scorer Self-Test ===\n")
    for t in test_cases:
        result = score_detection(**t, config=cfg)
        print(f"ID{result.track_id:2d} [{result.label:10s}] "
              f"conf={result.conf:.2f}  "
              f"score={result.threat_score:.3f}  "
              f"-> {result.threat_level.label():8s}  "
              f"zone={result.in_zone}  "
              f"| {', '.join(result.notes)}")
