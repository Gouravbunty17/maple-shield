"""Tests for scripts/benchmark.py.

We do not assert performance numbers here. We assert the contract: the
benchmark produces frames.jsonl + summary.json + manifest.json with the
documented schema, and CairnFrameRecord rows pass round-trip JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import run_benchmark  # noqa: E402


def test_benchmark_emits_frames_summary_and_manifest(tmp_path):
    out = tmp_path / "bench-test"
    summary = run_benchmark(output_dir=out, n_frames=20, n_tracks=2)

    assert (out / "frames.jsonl").exists()
    assert (out / "summary.json").exists()
    assert (out / "manifest.json").exists()

    assert summary["frames"] == 20
    assert summary["tracks_per_frame"] == 2
    assert summary["stage_measured"] == "cairn_scoring"
    assert "yolo_onnx_inference" in summary["stage_NOT_measured"]
    assert summary["frames_per_second_observed"] > 0
    for k in ("p50", "p95", "p99", "mean", "max"):
        assert k in summary["cairn_scoring_ms"]


def test_benchmark_jsonl_rows_are_valid_cairn_records(tmp_path):
    out = tmp_path / "bench-test-2"
    run_benchmark(output_dir=out, n_frames=5, n_tracks=1)

    rows = (out / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5
    for line in rows:
        row = json.loads(line)
        # CAIRN frame record contract
        assert row["schema"] == "cairn.frame.v1"
        assert "frame" in row and "ts" in row and "iso_time" in row
        assert "max_threat_level" in row and "max_risk_score" in row
        assert isinstance(row["risks"], list)
        # benchmark-specific addendum
        assert row["benchmark"]["stage_measured"] == "cairn_scoring"
        assert row["benchmark"]["cairn_scoring_ms"] >= 0


def test_benchmark_does_not_lie_about_yolo(tmp_path):
    """Make sure the benchmark does not pretend it measured YOLO inference."""
    out = tmp_path / "bench-honest"
    run_benchmark(output_dir=out, n_frames=3, n_tracks=1)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["stage_measured"] == "cairn_scoring"
    assert "yolo_onnx_inference" in summary["stage_NOT_measured"]
    assert "end_to_end_detection_to_alert" in summary["stage_NOT_measured"]
    assert "jetson_orin_nano" in summary["stage_NOT_measured"]
