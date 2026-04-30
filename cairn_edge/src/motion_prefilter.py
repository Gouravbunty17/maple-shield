"""CPU-side motion prefilter for Cairn-Edge.

This module decides which sky-region tiles deserve detector time. It is designed
for constrained Jetson deployments where running YOLO on every full frame wastes
power and latency budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class TileDecision:
    tile_id: str
    bbox: BBox
    motion_energy: float
    should_run_detector: bool


class MotionPrefilter:
    def __init__(self, min_motion_energy: float = 0.015, history: int = 120, var_threshold: float = 24.0) -> None:
        self.min_motion_energy = float(min_motion_energy)
        self._mog2 = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=False)

    @staticmethod
    def build_tiles(width: int, height: int, grid: str = "2x2", sky_roi: Sequence[float] = (0.0, 0.0, 1.0, 0.65)) -> List[Tuple[str, BBox]]:
        cols, rows = [int(part) for part in grid.lower().split("x", 1)]
        x0 = int(width * float(sky_roi[0]))
        y0 = int(height * float(sky_roi[1]))
        x1 = int(width * float(sky_roi[2]))
        y1 = int(height * float(sky_roi[3]))
        roi_w = max(1, x1 - x0)
        roi_h = max(1, y1 - y0)

        tiles: List[Tuple[str, BBox]] = []
        for r in range(rows):
            for c in range(cols):
                tx0 = x0 + int(c * roi_w / cols)
                ty0 = y0 + int(r * roi_h / rows)
                tx1 = x0 + int((c + 1) * roi_w / cols)
                ty1 = y0 + int((r + 1) * roi_h / rows)
                tiles.append((f"r{r}c{c}", (tx0, ty0, tx1, ty1)))
        return tiles

    def evaluate(self, frame_bgr: np.ndarray, tiles: Iterable[Tuple[str, BBox]]) -> List[TileDecision]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        fg = self._mog2.apply(gray)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        decisions: List[TileDecision] = []
        for tile_id, (x0, y0, x1, y1) in tiles:
            tile_mask = fg[y0:y1, x0:x1]
            if tile_mask.size == 0:
                energy = 0.0
            else:
                energy = float(np.count_nonzero(tile_mask)) / float(tile_mask.size)
            decisions.append(
                TileDecision(
                    tile_id=tile_id,
                    bbox=(x0, y0, x1, y1),
                    motion_energy=energy,
                    should_run_detector=energy >= self.min_motion_energy,
                )
            )
        return decisions
