"""edge-agent entry point.

Reads frames from a source, runs the detector, and posts DetectionFrame
packets to the fusion-engine.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone

import httpx

from edge_agent.detector import CairnMockDetector
from edge_agent.source import MockSource, VideoFileSource


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="mock",
                   help="'mock' or path to a video file")
    p.add_argument("--camera-id", default=os.environ.get("MAPLE_SHIELD_CAMERA_ID", "cam-01"))
    p.add_argument("--fusion", default=os.environ.get("MAPLE_SHIELD_FUSION", "http://localhost:8090"),
                   help="fusion-engine base URL")
    p.add_argument("--n-frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true",
                   help="don't actually POST; just print")
    args = p.parse_args()

    if args.source == "mock":
        src = MockSource(n_frames=args.n_frames)
    else:
        src = VideoFileSource(args.source)

    detector = CairnMockDetector()
    interval = 1.0 / max(0.1, args.fps)

    if args.dry_run:
        for i, frame in src.frames():
            h, w = frame.shape[:2]
            dets = detector.detect(i, w, h)
            print({"frame": i, "n_dets": len(dets),
                   "first_conf": dets[0].confidence if dets else None})
            time.sleep(interval)
        return

    with httpx.Client(timeout=2.0) as client:
        for i, frame in src.frames():
            h, w = frame.shape[:2]
            dets = detector.detect(i, w, h)
            ts = datetime.now(timezone.utc).timestamp()
            payload = {
                "frame_id": f"{args.camera_id}-{i}",
                "ts": ts,
                "camera_id": args.camera_id,
                "image_size": [w, h],
                "detections": [
                    {"cls": d.cls, "confidence": d.confidence, "bbox": list(d.bbox)}
                    for d in dets
                ],
            }
            try:
                client.post(f"{args.fusion}/detections", json=payload)
            except httpx.HTTPError as e:
                print(f"[edge-agent] fusion unreachable: {e}")
            time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
    main()
