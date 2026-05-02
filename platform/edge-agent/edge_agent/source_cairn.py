"""OpenCV frame source for CAIRN-backed edge-agent runs."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


def normalize_video_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    text = source.strip()
    if text.isdigit():
        return int(text)
    return source


class CairnFrameSource:
    """Yield ``(frame_idx, frame)`` from a webcam index or video path."""

    def __init__(self, source: str | int = 0, max_frames: Optional[int] = None):
        self.source = normalize_video_source(source)
        self.max_frames = max_frames

    def frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        try:
            import cv2  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "CairnFrameSource requires opencv-python-headless; pip install -r requirements.txt"
            ) from e

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open CAIRN frame source: {self.source}")

        i = 0
        try:
            while self.max_frames is None or i < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
        finally:
            cap.release()
