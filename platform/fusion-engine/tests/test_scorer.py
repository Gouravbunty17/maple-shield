"""Scorer threshold tests."""

from fusion.scorer import ScorerConfig, score_track
from fusion.tracker import Track


def _trk(n_obs=1, max_conf=0.5, last_conf=None) -> Track:
    return Track(
        track_id="t", camera_id="c", bbox=(0, 0, 10, 10),
        n_obs=n_obs, max_conf=max_conf,
        last_conf=last_conf if last_conf is not None else max_conf,
    )


def test_below_threshold_returns_no_alert():
    cfg = ScorerConfig(conf_threshold=0.6, dwell_min_obs=5)
    out = score_track(_trk(n_obs=1, max_conf=0.3), cfg)
    assert out is None


def test_single_observation_low_severity():
    cfg = ScorerConfig(conf_threshold=0.6, dwell_min_obs=5)
    out = score_track(_trk(n_obs=1, max_conf=0.7), cfg)
    assert out is not None
    assert out.severity == "low"
    assert out.rule == "single_obs"


def test_dwell_med_severity():
    cfg = ScorerConfig(conf_threshold=0.6, dwell_min_obs=5)
    out = score_track(_trk(n_obs=6, max_conf=0.7), cfg)
    assert out is not None
    assert out.severity == "med"
    assert out.rule == "dwell_over_threshold"


def test_persistent_high_severity():
    cfg = ScorerConfig()
    out = score_track(_trk(n_obs=15, max_conf=0.95), cfg)
    assert out is not None
    assert out.severity == "high"
    assert out.rule == "persistent_high_confidence"
