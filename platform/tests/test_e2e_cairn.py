"""End-to-end CAIRN adapter test with the platform contracts."""

from fastapi.testclient import TestClient

import app.main as cmd_main
from app.store import Store as CmdStore
from cairn_engine import CairnDetection, CairnEngine
from edge_agent.cairn_adapter import CairnSourceDetector
from fusion.scorer import score_track
from fusion.tracker import Tracker


def _cairn_provider(frame_idx: int, frame_w: int, frame_h: int):
    return [
        CairnDetection(
            track_id=42,
            label="drone",
            confidence=0.93,
            box=[10 + frame_idx, 20, 50 + frame_idx, 60],
            frame_w=frame_w,
            frame_h=frame_h,
            track_confirmed=True,
            persistence_frames=frame_idx + 1,
        )
    ]


def test_e2e_with_cairn_stub_creates_incident_and_preserves_track_id():
    cmd_main.store = CmdStore(":memory:")
    client = TestClient(cmd_main.app)

    detector = CairnSourceDetector(engine=CairnEngine(), detection_provider=_cairn_provider)
    tracker = Tracker(iou_thresh=0.1, max_misses=3)

    posted = 0
    for i in range(20):
        dets = detector.detect(i, 640, 360)
        tracker.update(
            "cam-1",
            float(i),
            [
                {
                    "cls": d.cls,
                    "confidence": d.confidence,
                    "bbox": list(d.bbox),
                    "track_id": d.track_id,
                }
                for d in dets
            ],
        )
        for trk in tracker.tracks.values():
            sa = score_track(trk)
            if sa is None:
                continue
            r = client.post(
                "/alerts",
                json={
                    "track_id": sa.track_id,
                    "camera_id": sa.camera_id,
                    "severity": sa.severity,
                    "rule": sa.rule,
                    "score": sa.score,
                },
            )
            assert r.status_code == 201
            posted += 1

    assert posted > 0
    assert detector.engine.frames_processed == 20
    assert list(tracker.tracks) == ["cairn-42"]

    incs = client.get("/incidents").json()
    assert len(incs) == 1
    assert incs[0]["status"] == "new"

    audit = client.get("/audit").json()
    assert audit["verified"] is True
    actions = {e["action"] for e in audit["entries"]}
    assert "incident.auto_create" in actions
