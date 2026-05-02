"""Tests for maple_shield_cairn_session.CairnSession."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cairn_engine import CairnEngine  # noqa: E402
from maple_shield_cairn_session import CairnSession  # noqa: E402


def test_session_writes_jsonl(tmp_path):
    jsonl = tmp_path / "frames.jsonl"
    with CairnSession(jsonl_path=str(jsonl), engine=CairnEngine()) as cairn:
        rec = cairn.process_dict_frame(
            frame_id=1,
            det_dicts=[
                {
                    "track_id": 1,
                    "label": "drone",
                    "confidence": 0.91,
                    "box": [100, 100, 140, 140],
                    "track_confirmed": True,
                    "persistence_frames": 12,
                }
            ],
            frame_w=640,
            frame_h=360,
            fps=30.0,
        )
    assert jsonl.exists()
    rows = jsonl.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["schema"] == "cairn.frame.v1"
    assert payload["frame"] == 1
    assert payload["risks"], "risks list should not be empty for a confident drone"
    risk = payload["risks"][0]
    assert "risk_score" in risk
    assert "threat_level" in risk
    assert "reasons" in risk
    assert "recommended_operator_action" in risk


def test_session_no_jsonl_when_not_requested(tmp_path):
    # No jsonl_path = no file written, but processing still works.
    cairn = CairnSession(engine=CairnEngine())
    rec = cairn.process_dict_frame(
        frame_id=1,
        det_dicts=[],
        frame_w=320,
        frame_h=240,
    )
    assert rec.frame == 1
    assert rec.risks == []
    # nothing should have been written under tmp_path
    assert list(tmp_path.iterdir()) == []
