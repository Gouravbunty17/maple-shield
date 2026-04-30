#!/usr/bin/env python3
"""Capture/calibrate RGB-to-thermal homography.

Usage:
    python scripts/calibrate_thermal.py --rgb 0 --thermal /dev/video1 --output cairn_edge/calibration/homography.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from cairn_edge.src.advanced.thermal_fusion import ThermalFusion, ThermalFusionConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate thermal/RGB homography using a chessboard target.")
    parser.add_argument("--rgb", default="0", help="RGB camera index or device path")
    parser.add_argument("--thermal", default="/dev/video1", help="Thermal camera index or device path")
    parser.add_argument("--output", default="cairn_edge/calibration/homography.npy")
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()

    rgb_source = int(args.rgb) if str(args.rgb).isdigit() else args.rgb
    thermal_source = int(args.thermal) if str(args.thermal).isdigit() else args.thermal
    rgb_cap = cv2.VideoCapture(rgb_source)
    th_cap = cv2.VideoCapture(thermal_source, cv2.CAP_V4L2)
    if not rgb_cap.isOpened():
        raise SystemExit(f"RGB camera unavailable: {args.rgb}")
    if not th_cap.isOpened():
        raise SystemExit(f"Thermal camera unavailable: {args.thermal}")

    ok_rgb, rgb_frame = rgb_cap.read()
    ok_th, th_frame = th_cap.read()
    if not ok_rgb or rgb_frame is None:
        raise SystemExit("failed to read RGB frame")
    if not ok_th or th_frame is None:
        raise SystemExit("failed to read thermal frame")

    fusion = ThermalFusion(ThermalFusionConfig(thermal_enabled=False))
    h = fusion.calibrate(rgb_frame, th_frame, chessboard_size=(args.cols, args.rows))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fusion.save_homography(out)
    print(f"saved homography to {out}")
    print(h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
