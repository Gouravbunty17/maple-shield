"""Tests for run_maple_shield_demo.py.

The demo wraps the benchmark and does PASS/FAIL gating. We assert the
gating works: a real run reports PASS and writes the expected files.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_runs_and_writes_outputs(tmp_path):
    out_dir = tmp_path / "demo-test"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_maple_shield_demo.py"),
            "--frames",
            "30",
            "--tracks",
            "1",
            "--output",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"demo failed: stdout={proc.stdout} stderr={proc.stderr}"
    assert "RESULT: PASS" in proc.stdout
    assert (out_dir / "frames.jsonl").exists()
    assert (out_dir / "summary.json").exists()

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 30
    assert summary["stage_measured"] == "cairn_scoring"


def test_demo_summary_shape_matches_documented_contract(tmp_path):
    out_dir = tmp_path / "demo-shape"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "run_maple_shield_demo.py"),
            "--frames",
            "10",
            "--output",
            str(out_dir),
        ],
        cwd=str(REPO_ROOT),
        check=True,
        timeout=60,
    )
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    for k in (
        "schema",
        "session_id",
        "started_iso",
        "ended_iso",
        "duration_s",
        "frames",
        "frames_per_second_observed",
        "stage_measured",
        "stage_NOT_measured",
        "cairn_scoring_ms",
    ):
        assert k in summary, f"summary missing key {k!r}"
