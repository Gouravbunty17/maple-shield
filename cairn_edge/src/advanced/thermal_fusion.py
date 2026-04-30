"""Thermal/RGB fusion for Cairn-Edge.

Designed for Jetson Orin Nano constraints:
- thermal path is optional and disabled unless configured
- hot-spot detection is simple OpenCV threshold + contours
- homography is loaded once and reused
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .models import HealthStatus

LOGGER = logging.getLogger("cairn_edge.thermal")


class ThermalDetection(BaseModel):
    """Thermal hot-spot detection aligned to RGB frame coordinates."""

    bbox: Tuple[int, int, int, int]
    temperature_c: float
    area_px: int = Field(ge=0)


class ThermalFusionConfig(BaseModel):
    thermal_enabled: bool = False
    camera_id: int | str = "/dev/video1"
    calibration_path: str = "cairn_edge/calibration/homography.npy"
    power_save: bool = True
    threshold_c: float = 50.0
    min_area_px: int = 100
    alpha: float = 0.4
    camera_width: int = 160
    camera_height: int = 120


class ThermalFusion:
    """Fuse an RGB frame with a secondary thermal camera frame."""

    def __init__(self, config: ThermalFusionConfig) -> None:
        self.config = config
        self.homography: Optional[np.ndarray] = self._load_homography(config.calibration_path)
        self.capture: Optional[cv2.VideoCapture] = None
        self._last_health = time.time()
        self._last_error_log = 0.0
        self._degraded_reason: Optional[str] = None
        if self.config.thermal_enabled:
            self.open()

    def open(self) -> None:
        if not self.config.thermal_enabled:
            return
        self.capture = cv2.VideoCapture(self.config.camera_id, cv2.CAP_V4L2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        if not self.capture.isOpened():
            self._mark_degraded("thermal camera unavailable")
            self.capture.release()
            self.capture = None

    @staticmethod
    def _load_homography(path: str) -> Optional[np.ndarray]:
        p = Path(path)
        if not p.exists():
            return None
        matrix = np.load(str(p))
        if matrix.shape != (3, 3):
            raise ValueError(f"invalid homography shape: {matrix.shape}")
        return matrix.astype(np.float32)

    def save_homography(self, path: str | Path) -> None:
        if self.homography is None:
            raise RuntimeError("no homography available to save")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), self.homography)

    def calibrate(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray, chessboard_size: Tuple[int, int] = (9, 6)) -> np.ndarray:
        """Estimate visible-to-thermal homography from chessboard corners."""
        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        thermal_gray = self._normalize_thermal_u8(thermal_frame)
        ok_rgb, corners_rgb = cv2.findChessboardCorners(rgb_gray, chessboard_size)
        ok_th, corners_th = cv2.findChessboardCorners(thermal_gray, chessboard_size)
        if not ok_rgb or not ok_th:
            raise RuntimeError("chessboard not found in both RGB and thermal frames")
        homography, mask = cv2.findHomography(corners_th.reshape(-1, 2), corners_rgb.reshape(-1, 2), cv2.RANSAC)
        if homography is None or mask is None:
            raise RuntimeError("homography estimation failed")
        self.homography = homography.astype(np.float32)
        return self.homography

    def read_thermal_frame(self) -> Optional[np.ndarray]:
        if not self.config.thermal_enabled:
            return None
        if self.capture is None:
            self.open()
        if self.capture is None:
            self._log_disconnect_once()
            return None
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._mark_degraded("thermal camera unavailable")
            self._log_disconnect_once()
            return None
        return frame

    def fuse(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, List[ThermalDetection]]:
        """Return fused RGB/thermal BGR frame and aligned thermal detections."""
        if not self.config.thermal_enabled:
            return rgb_frame, []
        thermal_frame = self.read_thermal_frame()
        if thermal_frame is None:
            return rgb_frame, []
        return self.fuse_with_thermal(rgb_frame, thermal_frame)

    def fuse_with_thermal(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray) -> Tuple[np.ndarray, List[ThermalDetection]]:
        thermal_u8 = self._normalize_thermal_u8(thermal_frame)
        thermal_temp_c = self._estimate_temperature_c(thermal_frame)
        h, w = rgb_frame.shape[:2]
        if self.homography is not None:
            aligned_u8 = cv2.warpPerspective(thermal_u8, self.homography, (w, h))
            aligned_temp = cv2.warpPerspective(thermal_temp_c.astype(np.float32), self.homography, (w, h))
        else:
            aligned_u8 = cv2.resize(thermal_u8, (w, h), interpolation=cv2.INTER_LINEAR)
            aligned_temp = cv2.resize(thermal_temp_c.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        thermal_color = cv2.applyColorMap(aligned_u8, cv2.COLORMAP_JET)
        fused = cv2.addWeighted(rgb_frame, 1.0 - self.config.alpha, thermal_color, self.config.alpha, 0)
        detections = self._detect_hotspots(aligned_temp)
        self._last_health = time.time()
        if detections or self._degraded_reason == "thermal camera unavailable":
            self._degraded_reason = None
        return fused, detections

    def _detect_hotspots(self, temp_c: np.ndarray) -> List[ThermalDetection]:
        mask = (temp_c > self.config.threshold_c).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[ThermalDetection] = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.config.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            roi = temp_c[y : y + h, x : x + w]
            detections.append(ThermalDetection(bbox=(x, y, w, h), temperature_c=float(np.max(roi)), area_px=area))
        return detections

    @staticmethod
    def _normalize_thermal_u8(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.dtype == np.uint8:
            return frame
        return cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    @staticmethod
    def _estimate_temperature_c(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        arr = frame.astype(np.float32)
        if frame.dtype == np.uint16:
            # Common radiometric convention: centikelvin. Fallback remains monotonic if vendor scale differs.
            return (arr / 100.0) - 273.15
        return (arr / 255.0) * 100.0

    def _mark_degraded(self, reason: str) -> None:
        self._degraded_reason = reason
        self._last_health = time.time()

    def _log_disconnect_once(self) -> None:
        now = time.time()
        if now - self._last_error_log >= 60.0:
            LOGGER.error("Thermal camera unavailable; continuing RGB-only")
            self._last_error_log = now

    def health(self) -> HealthStatus:
        return HealthStatus(
            module_name="thermal_fusion",
            status="degraded" if self._degraded_reason else "ok",
            last_heartbeat=self._last_health,
            degraded_reason=self._degraded_reason,
        )
