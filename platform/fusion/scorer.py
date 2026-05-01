"""Alert scoring.

The scorer is decision support: it labels tracks by severity according to
configurable rules, and the operator decides what to do next. The scorer
NEVER initiates an action against a target.

Rules implemented in MVP:
  * single_obs: low severity for a single confident detection
  * dwell_over_threshold: track has >= dwell_min_obs observations and
    max confidence >= conf_threshold
  * persistent_high_confidence: medium/high for sustained high-confidence
    presence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fusion.tracker import Track


@dataclass
class ScorerConfig:
    conf_threshold: float = 0.6
    dwell_min_obs: int = 5
    persistent_min_obs: int = 12
    persistent_min_max_conf: float = 0.85


SEVERITIES = ("info", "low", "med", "high")


@dataclass
class ScoredAlert:
    track_id: str
    camera_id: str
    rule: str
    severity: str
    score: float


def score_track(trk: Track, cfg: ScorerConfig = ScorerConfig()) -> Optional[ScoredAlert]:
    if trk.n_obs >= cfg.persistent_min_obs and trk.max_conf >= cfg.persistent_min_max_conf:
        return ScoredAlert(
            track_id=trk.track_id, camera_id=trk.camera_id,
            rule="persistent_high_confidence", severity="high",
            score=min(1.0, trk.max_conf),
        )
    if trk.n_obs >= cfg.dwell_min_obs and trk.max_conf >= cfg.conf_threshold:
        return ScoredAlert(
            track_id=trk.track_id, camera_id=trk.camera_id,
            rule="dwell_over_threshold", severity="med",
            score=min(1.0, trk.max_conf * 0.9),
        )
    if trk.last_conf >= cfg.conf_threshold:
        return ScoredAlert(
            track_id=trk.track_id, camera_id=trk.camera_id,
            rule="single_obs", severity="low",
            score=min(1.0, trk.last_conf * 0.6),
        )
    return None
