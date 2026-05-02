"""Video source abstraction.

Two sources are supported:
  * MockSource — synthetic frames; no codec dependency. Ideal for CI and
    the demo so the platform can run anywhere.
  * VideoFileSource — OpenCV-backed reader for an MP4/AVI on disk.

A real camera source is intentionally NOT included in the MVP because
deployment-specific wiring (RTSP, GigE, etc.) belongs to the integrator.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


class MockSource:
    def __init__(self, n_frames: int = 200, w: int = 640, h: int = 360):
        self.n = n_frames
        self.w, self.h = w, h

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        for i in range(self.n):
            # gray frame, no real imagery — this is a stub
            frame = np.full((self.h, self.w, 3), 24, dtype=np.uint8)
            yield i, frame


class VideoFileSource:
    def __init__(self, path: str):
        self.path = path

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        try:
            import cv2  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "VideoFileSource requires opencv-python-headless; pip install -r requirements.txt"
            ) from e
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
        finally:
            cap.release()
