import textwrap
import time

from cairn_edge.src.advanced.geofence_engine import GeofenceEngine, point_in_polygon
from cairn_edge.src.advanced.models import Track


def write_cfg(tmp_path):
    cfg = tmp_path / "zones.yaml"
    cfg.write_text(textwrap.dedent("""
    zones:
      - name: Test Zone
        polygon:
          - [43.6900, -79.6300]
          - [43.6900, -79.6200]
          - [43.6800, -79.6200]
          - [43.6800, -79.6300]
        altitude_min_m: 0
        altitude_max_m: 500
        action: alert
        risk_multiplier: 2.0
    """), encoding="utf-8")
    return cfg


def test_point_in_polygon():
    poly = [(43.69, -79.63), (43.69, -79.62), (43.68, -79.62), (43.68, -79.63)]
    assert point_in_polygon(43.685, -79.625, poly)
    assert not point_in_polygon(43.700, -79.625, poly)


def test_evaluate_track_uses_multiplier_and_cap(tmp_path):
    engine = GeofenceEngine(write_cfg(tmp_path))
    track = Track(track_id="a", lat=43.685, lon=-79.625, alt=100, confidence=0.9, class_id="uas", kinematic_risk=60, timestamp=time.time())
    risk = engine.evaluate_track(track)
    assert risk.score == 100
    assert risk.action == "alert"
    assert risk.zone_name == "Test Zone"


def test_crossing_detection(tmp_path):
    engine = GeofenceEngine(write_cfg(tmp_path))
    prev = Track(track_id="a", lat=43.700, lon=-79.625, alt=100, confidence=0.9, class_id="uas", kinematic_risk=10, timestamp=time.time())
    curr = Track(track_id="a", lat=43.685, lon=-79.625, alt=100, confidence=0.9, class_id="uas", kinematic_risk=10, timestamp=time.time())
    assert engine.check_crossing(prev, curr) == ("Test Zone", "entering")


def test_invalid_config_degrades(tmp_path):
    bad = tmp_path / "missing.yaml"
    engine = GeofenceEngine(bad)
    status = engine.health()
    assert status.status == "degraded"
    risk = engine.evaluate_track(Track(track_id="x", lat=0, lon=0, confidence=0.1, class_id="unknown", kinematic_risk=5, timestamp=time.time()))
    assert risk.action == "monitor"
