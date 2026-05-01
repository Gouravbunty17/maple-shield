"""End-to-end test: synthetic detections through fusion -> command-api.

Uses the in-process FastAPI test client for command-api and pure-Python
calls into fusion-engine's tracker/scorer. We avoid spinning real HTTP
servers; the contract is tested by `command-api/tests/test_api.py` and
`fusion-engine/tests/*`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "command-api"))
sys.path.insert(0, str(ROOT / "fusion-engine"))
sys.path.insert(0, str(ROOT / "edge-agent"))

from fastapi.testclient import TestClient

import app.main as cmd_main
from app.store import Store as CmdStore
from fusion.scorer import score_track
from fusion.tracker import Tracker


def test_full_flow_creates_incident_and_logs_audit():
    cmd_main.store = CmdStore(":memory:")
    client = TestClient(cmd_main.app)

    tracker = Tracker(iou_thresh=0.1, max_misses=3)

    # walk a high-confidence drone across many frames so we hit 'persistent_high_confidence'
    posted = 0
    for i in range(20):
        tracker.update("cam-1", float(i),
                       [{"cls": "drone", "confidence": 0.92,
                         "bbox": [10 + i, 10, 30 + i, 30]}])
        for trk in tracker.tracks.values():
            sa = score_track(trk)
            if sa is None:
                continue
            r = client.post("/alerts", json={
                "track_id": sa.track_id, "camera_id": sa.camera_id,
                "severity": sa.severity, "rule": sa.rule, "score": sa.score,
            })
            assert r.status_code == 201
            posted += 1

    assert posted > 0
    incs = client.get("/incidents").json()
    assert len(incs) >= 1
    assert any(i["status"] == "new" for i in incs)

    audit = client.get("/audit").json()
    assert audit["verified"] is True
    actions = {e["action"] for e in audit["entries"]}
    assert "incident.auto_create" in actions

    # exportable incident summary
    inc_id = incs[0]["incident_id"]
    r = client.get(f"/incidents/{inc_id}/export")
    assert r.status_code == 200
    body = r.json()
    assert body["summary_md"].startswith(f"# Incident {inc_id}")
