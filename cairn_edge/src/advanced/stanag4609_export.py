"""STANAG 4609 evidence export scaffold with MISB ST 0601 KLV metadata.

This exporter is intentionally not on the frame hot path. It accepts frame and
track snapshots at <=1 Hz, builds a compact KLV metadata record, and sends work
to a background thread.

Pipeline examples for deployment validation:

GStreamer example:
    appsrc name=video_src is-live=true format=time ! videoconvert ! x265enc tune=zerolatency speed-preset=ultrafast ! video/x-h265 ! mpegtsmux name=mux ! filesink location=output.ts
    appsrc name=klv_src is-live=true format=time caps=meta/x-klv,parsed=true ! mux.

FFmpeg example:
    ffmpeg -y -f rawvideo -pix_fmt bgr24 -s 1920x1080 -r 1 -i - -c:v libx265 -preset ultrafast -tune zerolatency -f mpegts output.ts

Note: true in-band KLV requires platform-specific mux support. This class builds
MISB-style local-set metadata payloads and provides the async export boundary;
final conformance must be validated with ffmpeg/exiftool on deployment hardware.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .models import HealthStatus, Track

MISB_LOCAL_SET_KEY = bytes.fromhex("060E2B34020B01010E01030101000000")


class KLVFrame(BaseModel):
    timestamp_ns: int
    mandatory_tags: Dict[int, Any]
    warning_flags: List[str] = Field(default_factory=list)


class PlatformState(BaseModel):
    camera_lat: float = 0.0
    camera_lon: float = 0.0
    camera_alt_m: float = 0.0
    platform_heading_deg: float = 0.0
    sensor_relative_azimuth_deg: float = 0.0
    sensor_relative_elevation_deg: float = 0.0
    frame_corners: List[Tuple[float, float]] = Field(default_factory=list)
    precision_time_available: bool = False


class STANAG4609Exporter:
    """Asynchronous evidence exporter for periodic video/KLV snapshots."""

    def __init__(self, output_dir: str | Path, platform_state: Optional[PlatformState] = None, enabled: bool = True, retry_limit: int = 3, queue_size: int = 8) -> None:
        self.output_dir = Path(output_dir)
        self.platform_state = platform_state or PlatformState()
        self.enabled = enabled
        self.retry_limit = retry_limit
        self.metadata_ring: Deque[KLVFrame] = deque(maxlen=30)
        self._queue: queue.Queue[Tuple[np.ndarray, List[Track], float]] = queue.Queue(maxsize=queue_size)
        self._failures = 0
        self._last_health = time.time()
        self._degraded_reason: Optional[str] = None
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="stanag4609-exporter", daemon=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if enabled:
            self._worker.start()

    def add_metadata(self, klv_dict: Dict[int, Any]) -> KLVFrame:
        frame = KLVFrame(timestamp_ns=self._timestamp_ns(), mandatory_tags=klv_dict, warning_flags=self._warning_flags())
        self.metadata_ring.append(frame)
        return frame

    def build_klv_frame(self, tracks: Iterable[Track]) -> KLVFrame:
        track_list = list(tracks)
        best = max(track_list, key=lambda t: t.kinematic_risk, default=None)
        tags: Dict[int, Any] = {
            2: self._timestamp_ns(),
            13: self._frame_corners(),
            14: {"lat": self.platform_state.camera_lat, "lon": self.platform_state.camera_lon, "alt_m": self.platform_state.camera_alt_m},
            15: self.platform_state.platform_heading_deg,
            65: {"azimuth_deg": self.platform_state.sensor_relative_azimuth_deg, "elevation_deg": self.platform_state.sensor_relative_elevation_deg},
        }
        if best is not None:
            tags[17] = self._estimate_slant_range(best)
            tags[101] = {"track_id": best.track_id, "class_id": best.class_id, "risk": best.kinematic_risk}
        return KLVFrame(timestamp_ns=tags[2], mandatory_tags=tags, warning_flags=self._warning_flags())

    def write_frame(self, frame: np.ndarray, tracks: Iterable[Track]) -> bool:
        """Queue a periodic export job. Intended call rate: <=1 Hz."""
        if not self.enabled:
            return False
        try:
            self._queue.put_nowait((frame.copy(), list(tracks), time.time()))
            return True
        except queue.Full:
            self._degraded_reason = "STANAG export queue full; dropping snapshot"
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame, tracks, ts = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                klv_frame = self.build_klv_frame(tracks)
                self.metadata_ring.append(klv_frame)
                self._encode_snapshot(frame, klv_frame, ts)
                self._failures = 0
                self._degraded_reason = None
                self._last_health = time.time()
            except Exception as exc:
                self._failures += 1
                self._degraded_reason = f"STANAG export failed: {exc}"
                if self._failures >= self.retry_limit:
                    self.enabled = False
                    self._degraded_reason = f"STANAG export disabled after {self._failures} failures: {exc}"

    def _encode_snapshot(self, frame: np.ndarray, klv_frame: KLVFrame, ts: float) -> None:
        height, width = frame.shape[:2]
        stem = time.strftime("evidence_%Y%m%dT%H%M%SZ", time.gmtime(ts))
        out_path = self.output_dir / f"{stem}.ts"
        klv_path = self.output_dir / f"{stem}.klv.json"
        klv_path.write_text(json.dumps(self._klv_json(klv_frame), indent=2, sort_keys=True), encoding="utf-8")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", "1",
            "-i", "-",
            "-frames:v", "1",
            "-c:v", "libx265",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-f", "mpegts",
            str(out_path),
        ]
        subprocess.run(cmd, input=frame.tobytes(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _timestamp_ns(self) -> int:
        return time.time_ns()

    def _warning_flags(self) -> List[str]:
        return [] if self.platform_state.precision_time_available else ["precision_time_unavailable_system_time_used"]

    def _frame_corners(self) -> List[Tuple[float, float]]:
        if len(self.platform_state.frame_corners) == 4:
            return self.platform_state.frame_corners
        lat = self.platform_state.camera_lat
        lon = self.platform_state.camera_lon
        d = 0.0005
        return [(lat + d, lon - d), (lat + d, lon + d), (lat - d, lon + d), (lat - d, lon - d)]

    def _estimate_slant_range(self, track: Track) -> float:
        # Lightweight approximation placeholder; deployment should use calibrated range estimator.
        alt_delta = max(0.0, track.alt - self.platform_state.camera_alt_m)
        return max(1.0, alt_delta)

    @staticmethod
    def encode_ber_length(length: int) -> bytes:
        if length < 0x80:
            return bytes([length])
        raw = length.to_bytes((length.bit_length() + 7) // 8, "big")
        return bytes([0x80 | len(raw)]) + raw

    def build_local_set_bytes(self, klv_frame: KLVFrame) -> bytes:
        payload = bytearray()
        for tag, value in sorted(klv_frame.mandatory_tags.items()):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
            payload.append(int(tag) & 0xFF)
            payload.extend(self.encode_ber_length(len(encoded)))
            payload.extend(encoded)
        payload.append(250)
        flags = json.dumps(klv_frame.warning_flags).encode("utf-8")
        payload.extend(self.encode_ber_length(len(flags)))
        payload.extend(flags)
        return MISB_LOCAL_SET_KEY + self.encode_ber_length(len(payload)) + bytes(payload)

    def _klv_json(self, klv_frame: KLVFrame) -> Dict[str, Any]:
        return {
            "timestamp_ns": klv_frame.timestamp_ns,
            "mandatory_tags": klv_frame.mandatory_tags,
            "warning_flags": klv_frame.warning_flags,
            "klv_hex_preview": self.build_local_set_bytes(klv_frame).hex()[:256],
        }

    def stop(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def health(self) -> HealthStatus:
        status = "ok"
        if not self.enabled and self._failures >= self.retry_limit:
            status = "error"
        elif self._degraded_reason:
            status = "degraded"
        return HealthStatus(module_name="stanag4609_export", status=status, last_heartbeat=self._last_health, degraded_reason=self._degraded_reason)
