import time

import numpy as np

from cairn_edge.src.advanced.models import Track
from cairn_edge.src.advanced.stanag4609_export import KLVFrame, PlatformState, STANAG4609Exporter


def test_klv_tag_building_and_timestamp_format(tmp_path):
    state = PlatformState(camera_lat=43.68, camera_lon=-79.62, camera_alt_m=100, platform_heading_deg=45, precision_time_available=False)
    exporter = STANAG4609Exporter(tmp_path, platform_state=state, enabled=False)
    track = Track(track_id="t1", lat=43.681, lon=-79.621, alt=130, velocity=10, heading=90, confidence=0.9, class_id="uas", kinematic_risk=75, timestamp=time.time())
    klv = exporter.build_klv_frame([track])
    assert isinstance(klv.timestamp_ns, int)
    assert klv.timestamp_ns > 1_000_000_000
    assert 2 in klv.mandatory_tags
    assert 13 in klv.mandatory_tags
    assert 14 in klv.mandatory_tags
    assert 15 in klv.mandatory_tags
    assert 17 in klv.mandatory_tags
    assert 65 in klv.mandatory_tags
    assert "precision_time_unavailable" in klv.warning_flags[0]


def test_local_set_bytes_have_key_and_payload(tmp_path):
    exporter = STANAG4609Exporter(tmp_path, enabled=False)
    klv = KLVFrame(timestamp_ns=123, mandatory_tags={2: 123, 13: [(1, 2), (1, 3), (0, 3), (0, 2)], 14: {"lat": 1, "lon": 2}, 15: 90, 65: {"azimuth_deg": 0, "elevation_deg": 0}})
    data = exporter.build_local_set_bytes(klv)
    assert data.startswith(bytes.fromhex("060E2B34020B01010E01030101000000"))
    assert len(data) > 20


def test_export_failure_disables_after_retries(tmp_path, monkeypatch):
    exporter = STANAG4609Exporter(tmp_path, enabled=False, retry_limit=3)

    def boom(frame, klv, ts):
        raise PermissionError("disk unavailable")

    monkeypatch.setattr(exporter, "_encode_snapshot", boom)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(3):
        try:
            exporter._encode_snapshot(frame, exporter.build_klv_frame([]), time.time())
        except PermissionError as exc:
            exporter._failures += 1
            exporter._degraded_reason = f"STANAG export failed: {exc}"
            if exporter._failures >= exporter.retry_limit:
                exporter.enabled = False
                exporter._degraded_reason = f"STANAG export disabled after {exporter._failures} failures: {exc}"
    status = exporter.health()
    assert status.status == "error"
    assert "disabled" in (status.degraded_reason or "")
