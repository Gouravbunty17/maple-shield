"""
Standalone CAIRN smoke demo.

This does not require a camera, ONNX Runtime, or OpenCV. It proves that the
CAIRN engine package can score tracks and produce audit-ready JSON records.

Usage:
    python cairn_demo.py
    python cairn_demo.py --config configs/cairn.default.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cairn_engine import CairnDetection, CairnEngine, CairnEngineConfig


def build_demo_detections(frame_w: int, frame_h: int) -> list[CairnDetection]:
    return [
        CairnDetection(
            track_id=101,
            label="drone",
            confidence=0.88,
            box=[600, 310, 690, 365],
            frame_w=frame_w,
            frame_h=frame_h,
            track_confirmed=True,
            velocity_px=12.4,
            vx=-8.0,
            vy=9.4,
            persistence_frames=36,
        ),
        CairnDetection(
            track_id=102,
            label="bird",
            confidence=0.61,
            box=[130, 120, 170, 150],
            frame_w=frame_w,
            frame_h=frame_h,
            track_confirmed=True,
            velocity_px=5.2,
            vx=5.0,
            vy=1.4,
            persistence_frames=4,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN Engine smoke demo")
    parser.add_argument("--config", default=None, help="Optional CAIRN config JSON path")
    args = parser.parse_args()

    config = CairnEngineConfig.from_json(args.config) if args.config else CairnEngineConfig()
    engine = CairnEngine(config)

    frame_w, frame_h = 1280, 720
    record = engine.process_frame(
        frame_id=1,
        detections=build_demo_detections(frame_w, frame_h),
        session_id=f"cairn-demo-{int(time.time())}",
        frame_w=frame_w,
        frame_h=frame_h,
        fps=25.0,
        infer_ms=None,
        health={"mode": "smoke_demo", "camera": "not_required"},
    )

    print(json.dumps(record.to_json(), indent=2))


if __name__ == "__main__":
    main()
