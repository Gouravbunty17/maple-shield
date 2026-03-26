"""
maple_shield_dashboard.py — Replay Dashboard for Maple Shield

Interactive web dashboard to replay and analyse Maple Shield detection runs.

Features:
  - Browse all saved runs
  - Radar-style track visualisation with threat colour coding
  - Timeline scrubber with play / pause
  - Alert history log
  - Per-run statistics

Usage:
    python maple_shield_dashboard.py
    # → http://localhost:5000

Requirements: pip install flask
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from flask import Flask, render_template, jsonify, abort, request

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__, template_folder="templates")

RUNS_DIR = Path("runs")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_runs() -> list[dict]:
    """Return sorted list of available runs with summary metadata."""
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        jsonl = run_dir / "detections.jsonl"
        meta  = run_dir / "meta.json"
        if not jsonl.exists():
            continue
        summary = {
            "name":        run_dir.name,
            "path":        str(run_dir),
            "frames":      0,
            "detections":  0,
            "max_threat":  "CLEAR",
            "duration_s":  0.0,
            "model":       "unknown",
            "provider":    "unknown",
            "has_video":   (run_dir / "overlay.mp4").exists(),
        }
        if meta.exists():
            try:
                m = json.loads(meta.read_text())
                summary["model"]    = Path(m.get("model", "")).name
                summary["provider"] = m.get("provider", "unknown")
            except Exception:
                pass
        # Quick scan for stats
        threat_rank = {"CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        t_start = t_end = None
        max_rank = 0
        n_det = 0
        try:
            with open(jsonl, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    summary["frames"] += 1
                    n_det += len(r.get("detections", []))
                    t = r.get("ts", 0)
                    if t_start is None or t < t_start: t_start = t
                    if t_end is None or t > t_end:     t_end   = t
                    rank = threat_rank.get(r.get("max_threat", "CLEAR"), 0)
                    if rank > max_rank:
                        max_rank = rank
                        summary["max_threat"] = r.get("max_threat", "CLEAR")
        except Exception:
            pass
        summary["detections"] = n_det
        if t_start and t_end:
            summary["duration_s"] = round(t_end - t_start, 1)
        runs.append(summary)
    return runs


def _load_run(run_name: str) -> list[dict]:
    """Load full JSONL for a run, return list of frame records."""
    path = RUNS_DIR / run_name / "detections.jsonl"
    if not path.exists():
        abort(404, description=f"Run not found: {run_name}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/runs")
def api_runs():
    return jsonify(_list_runs())


@app.route("/api/run/<run_name>")
def api_run(run_name: str):
    """Return all frame records for a run. Supports ?start=N&end=N for slicing."""
    records = _load_run(run_name)
    start = int(request.args.get("start", 0))
    end   = int(request.args.get("end",   len(records)))
    return jsonify(records[start:end])


@app.route("/api/run/<run_name>/summary")
def api_run_summary(run_name: str):
    """Return alerts (threat >= MEDIUM) and stats for a run."""
    records = _load_run(run_name)
    alerts = []
    threat_timeline = []
    threat_rank = {"CLEAR": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    for r in records:
        threat_timeline.append({
            "frame":  r.get("frame", 0),
            "ts":     r.get("ts", 0),
            "threat": r.get("max_threat", "CLEAR"),
            "rank":   threat_rank.get(r.get("max_threat", "CLEAR"), 0),
        })
        for det in r.get("detections", []):
            if threat_rank.get(det.get("threat", "CLEAR"), 0) >= 2:   # MEDIUM+
                alerts.append({
                    "frame":      r.get("frame", 0),
                    "ts":         r.get("ts", 0),
                    "track_id":   det.get("track_id"),
                    "label":      det.get("label"),
                    "threat":     det.get("threat"),
                    "score":      det.get("score"),
                    "conf":       det.get("conf"),
                    "box":        det.get("box"),
                    "in_zone":    det.get("in_zone"),
                    "velocity":   det.get("velocity_px"),
                })

    return jsonify({
        "run":             run_name,
        "total_frames":    len(records),
        "total_alerts":    len(alerts),
        "threat_timeline": threat_timeline,
        "alerts":          alerts,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maple Shield Replay Dashboard")
    parser.add_argument("--port",     default=5000, type=int)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    RUNS_DIR = Path(args.runs_dir)

    print(f"\n🛡️  Maple Shield Replay Dashboard")
    print(f"   Web UI   → http://localhost:{args.port}")
    print(f"   Runs dir → {RUNS_DIR.resolve()}")
    print(f"   Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)
