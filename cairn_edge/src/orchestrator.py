"""Cairn-Edge orchestrator.

Goal for the next fieldable milestone:
- run multiple camera streams through one measurable processing loop
- keep every advanced module behind explicit feature flags
- record latency and module health on every iteration
- avoid blocking the hot path with storage, export, or network operations

This file intentionally provides integration glue and safe defaults. Hardware
specific camera/detector/tracker implementations can be injected by the runtime
entrypoint or tests.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import numpy as np

from cairn_edge.src.advanced.models import Detection, HealthStatus, Track

LOGGER = logging.getLogger("cairn_edge.orchestrator")


class FrameSource(Protocol):
    stream_id: str

    def read(self) -> Optional[np.ndarray]:
        """Return the latest BGR frame or None if unavailable."""


class Detector(Protocol):
    def predict(self, frames: Mapping[str, np.ndarray]) -> Dict[str, List[Detection]]:
        """Return detections per stream id."""


class Tracker(Protocol):
    def update(self, detections: Mapping[str, List[Detection]]) -> List[Track]:
        """Return current tracks after detector association."""


@dataclass
class CairnEdgeFeatureFlags:
    enable_adversarial: bool = False
    enable_thermal: bool = False
    enable_swarm: bool = False
    enable_geofence: bool = False
    enable_mesh: bool = False
    enable_stanag: bool = False
    enable_hard_examples: bool = False
    enable_update_loader: bool = False


@dataclass
class CairnEdgeRuntimeConfig:
    target_latency_ms: float = 400.0
    log_path: str = "logs/cairn_edge_latency.jsonl"
    stanag_period_frames: int = 30
    thermal_period_frames: int = 5
    update_check_period_s: float = 3600.0
    max_loop_sleep_s: float = 0.001
    flags: CairnEdgeFeatureFlags = field(default_factory=CairnEdgeFeatureFlags)


class HealthMonitor:
    """Collect health states from optional Cairn-Edge modules."""

    def __init__(self) -> None:
        self.status: Dict[str, HealthStatus] = {}

    def update(self, module_name: str, health: HealthStatus) -> None:
        self.status[module_name] = health
        if health.status != "ok":
            LOGGER.warning("%s health=%s reason=%s", module_name, health.status, health.degraded_reason)

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, health in self.status.items():
            out[name] = health.model_dump() if hasattr(health, "model_dump") else health.dict()
        return out


class LatencyLogger:
    """Append loop metrics to JSONL for soak tests."""

    def __init__(self, path: str | Path, target_latency_ms: float) -> None:
        self.path = Path(path)
        self.target_latency_ms = target_latency_ms
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        record["over_budget"] = record.get("latency_ms", 0.0) > self.target_latency_ms
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


class CairnEdgeProcessor:
    """Main edge loop orchestrating detection, tracking, and optional modules."""

    def __init__(
        self,
        frame_sources: Sequence[FrameSource],
        detector: Detector,
        tracker: Tracker,
        config: Optional[CairnEdgeRuntimeConfig] = None,
        adversarial_detector: Any = None,
        thermal_fusion: Any = None,
        swarm_clusterer: Any = None,
        geofence_engine: Any = None,
        mesh_sync: Any = None,
        stanag_exporter: Any = None,
        hard_examples: Any = None,
        update_loader: Any = None,
        health_monitor: Optional[HealthMonitor] = None,
    ) -> None:
        self.frame_sources = list(frame_sources)
        self.detector = detector
        self.tracker = tracker
        self.config = config or CairnEdgeRuntimeConfig()
        self.adversarial_detector = adversarial_detector
        self.thermal_fusion = thermal_fusion
        self.swarm_clusterer = swarm_clusterer
        self.geofence_engine = geofence_engine
        self.mesh_sync = mesh_sync
        self.stanag_exporter = stanag_exporter
        self.hard_examples = hard_examples
        self.update_loader = update_loader
        self.health_monitor = health_monitor or HealthMonitor()
        self.latency_logger = LatencyLogger(self.config.log_path, self.config.target_latency_ms)
        self.frame_index = 0
        self._last_update_check = 0.0

    def read_frames(self) -> Dict[str, np.ndarray]:
        frames: Dict[str, np.ndarray] = {}
        for source in self.frame_sources:
            frame = source.read()
            if frame is not None:
                frames[source.stream_id] = frame
        return frames

    def step(self, operator_feedback: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        self.frame_index += 1
        timings: Dict[str, float] = {}

        frames = self._timed("read_frames", timings, self.read_frames)
        if not frames:
            record = self._record(t0, timings, 0, 0, 0)
            self.latency_logger.write(record)
            return record

        fused_frames = dict(frames)
        thermal_detections: Dict[str, list[Any]] = {}
        if self.config.flags.enable_thermal and self.thermal_fusion and self.frame_index % self.config.thermal_period_frames == 1:
            def fuse_all() -> None:
                for stream_id, frame in frames.items():
                    fused, detections = self.thermal_fusion.fuse(frame)
                    fused_frames[stream_id] = fused
                    thermal_detections[stream_id] = detections
            self._timed("thermal", timings, fuse_all)

        detections = self._timed("detect", timings, lambda: self.detector.predict(fused_frames))

        if self.config.flags.enable_adversarial and self.adversarial_detector:
            def run_adversarial() -> None:
                for stream_id, dets in list(detections.items()):
                    detections[stream_id] = self.adversarial_detector.process_detections(fused_frames[stream_id], dets)
            self._timed("adversarial", timings, run_adversarial)

        tracks = self._timed("track", timings, lambda: self.tracker.update(detections))

        module_outputs: Dict[str, Any] = {}
        if self.config.flags.enable_swarm and self.swarm_clusterer:
            module_outputs["swarm"] = self._timed("swarm", timings, lambda: self.swarm_clusterer.update(tracks))

        if self.config.flags.enable_geofence and self.geofence_engine:
            module_outputs["geofence"] = self._timed("geofence", timings, lambda: [self.geofence_engine.evaluate_track(track) for track in tracks])

        if self.config.flags.enable_mesh and self.mesh_sync:
            def mesh_step() -> None:
                for track in tracks:
                    self.mesh_sync.broadcast_track(track)
                for _ in range(4):
                    self.mesh_sync.receive_once()
            self._timed("mesh", timings, mesh_step)

        if self.config.flags.enable_stanag and self.stanag_exporter and self.frame_index % self.config.stanag_period_frames == 0:
            first_frame = next(iter(fused_frames.values()))
            self._timed("stanag_enqueue", timings, lambda: self.stanag_exporter.write_frame(first_frame, tracks))

        if self.config.flags.enable_hard_examples and self.hard_examples and operator_feedback:
            self._timed("hard_examples", timings, lambda: self._handle_operator_feedback(fused_frames, detections, operator_feedback))

        if self.config.flags.enable_update_loader and self.update_loader:
            now = time.time()
            if now - self._last_update_check >= self.config.update_check_period_s:
                self._last_update_check = now
                self._timed("update_check", timings, self.update_loader.check_for_updates)

        self._collect_health()
        record = self._record(t0, timings, len(frames), sum(len(v) for v in detections.values()), len(tracks))
        record["modules"] = list(module_outputs.keys())
        record["thermal_detection_count"] = sum(len(v) for v in thermal_detections.values())
        record["health"] = self.health_monitor.snapshot()
        self.latency_logger.write(record)
        return record

    def run_forever(self) -> None:
        while True:
            self.step()
            if self.config.max_loop_sleep_s > 0:
                time.sleep(self.config.max_loop_sleep_s)

    def _handle_operator_feedback(self, frames: Mapping[str, np.ndarray], detections: Mapping[str, List[Detection]], feedback: List[Dict[str, Any]]) -> None:
        by_stream = detections
        for item in feedback:
            stream_id = item.get("stream_id")
            detection_index = int(item.get("detection_index", -1))
            if stream_id not in frames or stream_id not in by_stream:
                continue
            if detection_index < 0 or detection_index >= len(by_stream[stream_id]):
                continue
            self.hard_examples.maybe_save(frames[stream_id], by_stream[stream_id][detection_index], item.get("operator_signatures", []))

    def _collect_health(self) -> None:
        modules = {
            "adversarial": self.adversarial_detector,
            "thermal": self.thermal_fusion,
            "swarm": self.swarm_clusterer,
            "geofence": self.geofence_engine,
            "mesh": self.mesh_sync,
            "stanag": self.stanag_exporter,
            "hard_examples": self.hard_examples,
        }
        for name, module in modules.items():
            if module is not None and hasattr(module, "health"):
                self.health_monitor.update(name, module.health())

    @staticmethod
    def _timed(name: str, timings: Dict[str, float], func: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        result = func()
        timings[f"{name}_ms"] = (time.perf_counter() - start) * 1000.0
        return result

    def _record(self, t0: float, timings: Dict[str, float], streams: int, detections: int, tracks: int) -> Dict[str, Any]:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "timestamp": time.time(),
            "frame_index": self.frame_index,
            "latency_ms": latency_ms,
            "timings": timings,
            "streams": streams,
            "detections": detections,
            "tracks": tracks,
        }
