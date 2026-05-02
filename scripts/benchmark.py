"""Maple Shield repeatable benchmark for the CAIRN risk-scoring stage.

Measures end-to-end latency of ``CairnEngine.process_frame`` on deterministic
synthetic detections. Produces:

  runs/<session_id>/frames.jsonl      one CairnFrameRecord per frame
  runs/<session_id>/summary.json      session-level metrics
  runs/<session_id>/manifest.json     environment / config snapshot

What this benchmark **does** measure
  * Per-call wall-time of ``CairnEngine.process_frame`` (the CAIRN scoring
    stage). Labelled ``cairn_scoring_ms`` in the summary.
  * Frames per second of that stage on the host CPU.
  * p50 / p95 / p99 of the per-call latency.
  * Total run duration.

What this benchmark **does NOT** measure
  * Real ONNX YOLO inference latency (that path needs a real camera or
    sample video and the model file).
  * End-to-end detection-to-alert wall time (no camera frame source here).
  * Jetson Orin Nano performance (run on the host this script is invoked on).

If you want a single number for the marketing line "30+ FPS on host CPU",
take ``summary.json::frames_per_second`` here. If you want a Jetson number,
run this script on a Jetson — the script does not lie about what it
measured.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --frames 600 --tracks 3
    python scripts/benchmark.py --output runs/my-bench
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

# Ensure the repo root is on sys.path when invoked as ``python scripts/benchmark.py``.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cairn_engine import (  # noqa: E402
    CairnDetection,
    CairnEngine,
    CairnEngineConfig,
)


def _output_ref(path: Path) -> str:
    """Return a readable output path without assuming it lives under the repo."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _build_synthetic_detections(
    frame_idx: int,
    n_tracks: int,
    frame_w: int,
    frame_h: int,
) -> List[CairnDetection]:
    """Produce deterministic synthetic CairnDetection rows for a frame.

    Trajectory is a simple linear walk so confidence and persistence climb
    with frame index, exercising the full risk ladder over a session.
    """
    out: List[CairnDetection] = []
    for k in range(n_tracks):
        x = (40 + (frame_idx * 6 + k * 80)) % max(1, frame_w - 80)
        y = 80 + ((k * 60) + (frame_idx % 40))
        w, h = 40, 30
        confidence = min(0.97, 0.55 + 0.005 * frame_idx + 0.05 * k)
        persistence = min(120, frame_idx)
        # All synthetic rows are drone-class — Maple Shield is drone-only.
        out.append(
            CairnDetection(
                track_id=100 + k,
                label="drone",
                confidence=float(confidence),
                box=[int(x), int(y), int(x + w), int(y + h)],
                frame_w=frame_w,
                frame_h=frame_h,
                track_confirmed=persistence > 8,
                velocity_px=6.0,
                vx=4.0,
                vy=2.0,
                persistence_frames=int(persistence),
            )
        )
    return out


def _percentile(samples: Iterable[float], p: float) -> float:
    s = sorted(samples)
    if not s:
        return 0.0
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    # nearest-rank
    rank = max(1, int(round(p / 100.0 * len(s))))
    return s[min(rank, len(s)) - 1]


def run_benchmark(
    output_dir: Path,
    n_frames: int = 300,
    n_tracks: int = 2,
    frame_w: int = 1280,
    frame_h: int = 720,
    target_fps: float = 30.0,
) -> dict:
    """Run the benchmark and write JSONL + summary.json + manifest.json.

    Returns the summary dict.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_path = output_dir / "frames.jsonl"
    summary_path = output_dir / "summary.json"
    manifest_path = output_dir / "manifest.json"

    engine = CairnEngine(CairnEngineConfig())
    session_id = output_dir.name

    per_frame_latencies_ms: List[float] = []
    total_risks = 0

    started_wall = time.time()
    started_iso = datetime.fromtimestamp(started_wall, tz=timezone.utc).isoformat()

    with frames_path.open("w", encoding="utf-8") as fh:
        for frame_idx in range(1, n_frames + 1):
            detections = _build_synthetic_detections(
                frame_idx, n_tracks, frame_w, frame_h
            )

            t0 = time.perf_counter()
            record = engine.process_frame(
                frame_id=frame_idx,
                detections=detections,
                session_id=session_id,
                frame_w=frame_w,
                frame_h=frame_h,
                fps=target_fps,
                infer_ms=None,  # we do not run a YOLO step here; do not fake it
                health={
                    "mode": "benchmark",
                    "stage_measured": "cairn_scoring",
                    "host_only": True,
                },
            )
            t1 = time.perf_counter()

            cairn_scoring_ms = (t1 - t0) * 1000.0
            per_frame_latencies_ms.append(cairn_scoring_ms)
            total_risks += len(record.risks)

            payload = record.to_json()
            payload["benchmark"] = {
                "cairn_scoring_ms": round(cairn_scoring_ms, 4),
                "stage_measured": "cairn_scoring",
            }
            fh.write(json.dumps(payload, sort_keys=True) + "\n")

    ended_wall = time.time()
    duration_s = max(1e-6, ended_wall - started_wall)
    fps_observed = n_frames / duration_s

    summary = {
        "schema": "maple_shield.benchmark.v1",
        "session_id": session_id,
        "started_iso": started_iso,
        "ended_iso": datetime.fromtimestamp(ended_wall, tz=timezone.utc).isoformat(),
        "duration_s": round(duration_s, 4),
        "frames": n_frames,
        "tracks_per_frame": n_tracks,
        "total_risks_emitted": total_risks,
        "frames_per_second_observed": round(fps_observed, 3),
        "frames_per_second_target": target_fps,
        "stage_measured": "cairn_scoring",
        "stage_NOT_measured": [
            "yolo_onnx_inference",
            "camera_capture",
            "end_to_end_detection_to_alert",
            "jetson_orin_nano",
        ],
        "cairn_scoring_ms": {
            "p50": round(_percentile(per_frame_latencies_ms, 50), 4),
            "p95": round(_percentile(per_frame_latencies_ms, 95), 4),
            "p99": round(_percentile(per_frame_latencies_ms, 99), 4),
            "mean": round(statistics.fmean(per_frame_latencies_ms), 4)
            if per_frame_latencies_ms
            else 0.0,
            "max": round(max(per_frame_latencies_ms), 4)
            if per_frame_latencies_ms
            else 0.0,
        },
        "frame_size": [frame_w, frame_h],
        "outputs": {
            "frames_jsonl": _output_ref(frames_path),
            "summary_json": _output_ref(summary_path),
            "manifest_json": _output_ref(manifest_path),
        },
    }

    manifest = {
        "schema": "maple_shield.benchmark.manifest.v1",
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "engine_metadata": engine.metadata(),
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Maple Shield CAIRN-scoring benchmark.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--tracks", type=int, default=2)
    parser.add_argument("--frame-w", type=int, default=1280)
    parser.add_argument("--frame-h", type=int, default=720)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to runs/bench-<UTC timestamp>.",
    )
    args = parser.parse_args()

    if args.output:
        out = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = _REPO_ROOT / "runs" / f"bench-{ts}"

    summary = run_benchmark(
        output_dir=out,
        n_frames=args.frames,
        n_tracks=args.tracks,
        frame_w=args.frame_w,
        frame_h=args.frame_h,
        target_fps=args.target_fps,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
