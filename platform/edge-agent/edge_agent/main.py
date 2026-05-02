"""edge-agent entry point.

Reads frames from a source, runs the detector, and posts DetectionFrame
packets to the fusion-engine.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

from edge_agent.detector import CairnMockDetector
from edge_agent.source import MockSource, VideoFileSource


def _detection_payload(d):
    payload = {"cls": d.cls, "confidence": d.confidence, "bbox": list(d.bbox)}
    if d.track_id:
        payload["track_id"] = d.track_id
    return payload


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="mock",
                   help="'mock' or path to a video file")
    p.add_argument("--detector", choices=["mock", "cairn-yolo"], default="mock",
                   help="detector backend; mock stays the repeatable CI default")
    p.add_argument("--yolo-model", default=os.environ.get("MAPLE_SHIELD_YOLO_MODEL"),
                   help="ONNX model path required when --detector cairn-yolo")
    p.add_argument("--yolo-classes", default=os.environ.get("MAPLE_SHIELD_YOLO_CLASSES"),
                   help="comma-separated labels or a newline-delimited label file for the YOLO model")
    p.add_argument("--conf-threshold", type=float, default=0.35,
                   help="minimum detector confidence for cairn-yolo")
    p.add_argument("--iou-threshold", type=float, default=0.45,
                   help="NMS and track association IoU threshold for cairn-yolo")
    p.add_argument("--max-detections", type=int, default=50,
                   help="maximum detections per frame for cairn-yolo")
    p.add_argument("--camera-id", default=os.environ.get("MAPLE_SHIELD_CAMERA_ID", "cam-01"))
    p.add_argument("--fusion", default=os.environ.get("MAPLE_SHIELD_FUSION", "http://localhost:8090"),
                   help="fusion-engine base URL")
    p.add_argument("--n-frames", type=int, default=120)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--dry-run", action="store_true",
                   help="don't actually POST; just print")
    return p


def _build_source(args):
    if args.source == "mock":
        return MockSource(n_frames=args.n_frames)
    return VideoFileSource(args.source)


def _load_class_names(spec: str | None):
    if not spec:
        return None
    path = Path(spec)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [part.strip() for part in spec.split(",") if part.strip()]


def _build_detector(args, provider_cls=None, cairn_detector_cls=None):
    if args.detector == "mock":
        return CairnMockDetector()

    if args.detector == "cairn-yolo":
        if not args.yolo_model:
            raise ValueError("--yolo-model is required when --detector cairn-yolo")

        if provider_cls is None:
            from edge_agent.yolo_provider import YoloDetectionProvider

            provider_cls = YoloDetectionProvider
        if cairn_detector_cls is None:
            from edge_agent.cairn_adapter import CairnSourceDetector

            cairn_detector_cls = CairnSourceDetector

        provider_kwargs = {
            "conf_threshold": args.conf_threshold,
            "iou_threshold": args.iou_threshold,
            "max_detections": args.max_detections,
        }
        class_names = _load_class_names(args.yolo_classes)
        if class_names is not None:
            provider_kwargs["class_names"] = class_names

        provider = provider_cls.from_onnx(args.yolo_model, **provider_kwargs)
        return cairn_detector_cls(detection_provider=provider, fps=args.fps)

    raise ValueError(f"unsupported detector: {args.detector}")


def _detect_frame(detector, frame_idx: int, frame, frame_w: int, frame_h: int):
    detect_frame = getattr(detector, "detect_frame", None)
    if callable(detect_frame):
        return detect_frame(frame_idx, frame)
    return detector.detect(frame_idx, frame_w, frame_h)


def main(argv: Sequence[str] | None = None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        src = _build_source(args)
        detector = _build_detector(args)
    except ValueError as e:
        parser.error(str(e))

    interval = 1.0 / max(0.1, args.fps)

    if args.dry_run:
        for i, frame in src.frames():
            h, w = frame.shape[:2]
            dets = _detect_frame(detector, i, frame, w, h)
            print({"frame": i, "n_dets": len(dets),
                   "first_conf": dets[0].confidence if dets else None})
            time.sleep(interval)
        return

    with httpx.Client(timeout=2.0, trust_env=False) as client:
        for i, frame in src.frames():
            h, w = frame.shape[:2]
            dets = _detect_frame(detector, i, frame, w, h)
            ts = datetime.now(timezone.utc).timestamp()
            payload = {
                "frame_id": f"{args.camera_id}-{i}",
                "ts": ts,
                "camera_id": args.camera_id,
                "image_size": [w, h],
                "detections": [_detection_payload(d) for d in dets],
            }
            try:
                client.post(f"{args.fusion}/detections", json=payload)
            except httpx.HTTPError as e:
                print(f"[edge-agent] fusion unreachable: {e}")
            time.sleep(interval)


if __name__ == "__main__":  # pragma: no cover
    main()
