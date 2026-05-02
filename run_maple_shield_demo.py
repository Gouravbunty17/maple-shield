"""One-command Maple Shield demo.

This script is intentionally small. It calls the CAIRN benchmark with
deterministic synthetic input (no camera, no model weights), verifies the
outputs, and prints a PASS / FAIL block with concrete numbers an operator or
reviewer can act on.

Outputs land under ``runs/demo-<UTC timestamp>/``:

    frames.jsonl     one CairnFrameRecord per frame
    summary.json     observed FPS, latency p50/p95/p99, frames, duration
    manifest.json    python / platform / engine metadata

Honest about scope:
  * Stage measured here is CAIRN risk scoring on host CPU.
  * YOLO inference, camera capture, Jetson timing, and end-to-end
    detection-to-alert latency are NOT measured by this demo. See
    docs/DEMO_VALIDATION.md for what counts as proven vs roadmap.

Usage:
    python run_maple_shield_demo.py
    python run_maple_shield_demo.py --frames 300 --tracks 2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# Ensure the repo root is on sys.path when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.benchmark import run_benchmark  # noqa: E402


# Pass criteria for the demo. These are intentionally generous because the
# repo runs on heterogeneous host CPUs. The test asserts the demo can keep
# up with a relaxed budget; tightening this is a roadmap item once we have
# Jetson numbers.
DEMO_MIN_FPS = 60.0  # CAIRN scoring stage on host CPU should clear this easily
DEMO_MAX_P95_MS = 50.0  # CAIRN scoring stage p95 budget on host CPU


def _check(summary: dict) -> Tuple[bool, list[str]]:
    """Return (overall_ok, reasons)."""
    reasons: list[str] = []

    if summary["frames"] <= 0:
        reasons.append("no frames recorded")

    if summary["frames_per_second_observed"] < DEMO_MIN_FPS:
        reasons.append(
            f"FPS {summary['frames_per_second_observed']:.1f} below floor {DEMO_MIN_FPS:.1f}"
        )

    p95 = float(summary["cairn_scoring_ms"]["p95"])
    if p95 > DEMO_MAX_P95_MS:
        reasons.append(
            f"CAIRN scoring p95 {p95:.2f} ms exceeds budget {DEMO_MAX_P95_MS:.2f} ms"
        )

    if summary["stage_measured"] != "cairn_scoring":
        reasons.append("stage_measured field is unexpected")

    return (len(reasons) == 0), reasons


def _validate_jsonl(frames_path: Path, expected_frames: int) -> Tuple[bool, str]:
    if not frames_path.exists():
        return False, f"missing {frames_path}"
    n = 0
    with frames_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                return False, f"invalid JSON on line {n + 1}: {e}"
            for key in ("schema", "frame", "ts", "session_id", "max_threat_level", "risks"):
                if key not in row:
                    return False, f"missing field {key!r} on line {n + 1}"
            n += 1
    if n != expected_frames:
        return False, f"expected {expected_frames} frames in JSONL, got {n}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Maple Shield one-command demo.")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--tracks", type=int, default=2)
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory. Defaults to runs/demo-<UTC timestamp>.",
    )
    args = parser.parse_args()

    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = _REPO_ROOT / "runs" / f"demo-{ts}"

    print(f"[demo] running CAIRN scoring benchmark, {args.frames} frames, {args.tracks} tracks")
    summary = run_benchmark(
        output_dir=out_dir,
        n_frames=args.frames,
        n_tracks=args.tracks,
    )

    frames_path = out_dir / "frames.jsonl"
    jsonl_ok, jsonl_msg = _validate_jsonl(frames_path, args.frames)
    pass_ok, reasons = _check(summary)
    if not jsonl_ok:
        pass_ok = False
        reasons.append(f"JSONL validation: {jsonl_msg}")

    print()
    print("=" * 60)
    print("Maple Shield demo report")
    print("=" * 60)
    print(f"output dir          : {out_dir}")
    print(f"frames              : {summary['frames']}")
    print(f"duration_s          : {summary['duration_s']:.3f}")
    print(f"observed FPS        : {summary['frames_per_second_observed']:.2f}")
    print(f"CAIRN p50 ms        : {summary['cairn_scoring_ms']['p50']:.3f}")
    print(f"CAIRN p95 ms        : {summary['cairn_scoring_ms']['p95']:.3f}")
    print(f"CAIRN p99 ms        : {summary['cairn_scoring_ms']['p99']:.3f}")
    print(f"total risks emitted : {summary['total_risks_emitted']}")
    print(f"stage measured      : {summary['stage_measured']}")
    print(
        "stage NOT measured  : "
        + ", ".join(summary["stage_NOT_measured"])
    )
    print(f"JSONL validation    : {jsonl_msg}")

    print()
    if pass_ok:
        print("RESULT: PASS")
        print(
            "(CAIRN scoring stage on host CPU. "
            "Real-camera + Jetson + end-to-end numbers remain roadmap; see "
            "docs/DEMO_VALIDATION.md.)"
        )
    else:
        print("RESULT: FAIL")
        for r in reasons:
            print(f"  - {r}")

    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
