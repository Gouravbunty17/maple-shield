import json
from pathlib import Path

import numpy as np

from cairn_edge.src.advanced.models import Detection, HealthStatus, Track
from cairn_edge.src.orchestrator import CairnEdgeFeatureFlags, CairnEdgeProcessor, CairnEdgeRuntimeConfig


class MockFrameSource:
    def __init__(self, stream_id):
        self.stream_id = stream_id

    def read(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)


class MockDetector:
    def predict(self, frames):
        return {stream_id: [Detection(bbox=(0, 0, 32, 32), confidence=0.95, class_id="uas")] for stream_id in frames}


class MockAdversarial:
    def process_detections(self, frame, detections):
        for det in detections:
            det.adversarial = True
            det.adversarial_score = 0.91
        return detections

    def health(self):
        return HealthStatus(module_name="adversarial", status="ok", last_heartbeat=1.0)


class MockTracker:
    def update(self, detections):
        tracks = []
        for stream_id, dets in detections.items():
            for i, det in enumerate(dets):
                tracks.append(Track(track_id=f"{stream_id}-{i}", lat=43.0, lon=-79.0, confidence=det.confidence, class_id=det.class_id, adversarial=det.adversarial, adversarial_score=det.adversarial_score))
        return tracks


class MockSwarm:
    def update(self, tracks):
        return []

    def health(self):
        return HealthStatus(module_name="swarm", status="ok", last_heartbeat=1.0)


class MockGeofence:
    def evaluate_track(self, track):
        return {"track_id": track.track_id, "action": "alert" if track.adversarial else "monitor"}

    def health(self):
        return HealthStatus(module_name="geofence", status="ok", last_heartbeat=1.0)


def test_orchestrator_sequence_and_latency_log(tmp_path):
    log_path = tmp_path / "latency.jsonl"
    cfg = CairnEdgeRuntimeConfig(
        log_path=str(log_path),
        flags=CairnEdgeFeatureFlags(enable_adversarial=True, enable_swarm=True, enable_geofence=True),
    )
    processor = CairnEdgeProcessor(
        frame_sources=[MockFrameSource("cam1"), MockFrameSource("cam2")],
        detector=MockDetector(),
        tracker=MockTracker(),
        config=cfg,
        adversarial_detector=MockAdversarial(),
        swarm_clusterer=MockSwarm(),
        geofence_engine=MockGeofence(),
    )
    record = processor.step()
    assert record["streams"] == 2
    assert record["detections"] == 2
    assert record["tracks"] == 2
    assert "detect_ms" in record["timings"]
    assert "track_ms" in record["timings"]
    assert "adversarial_ms" in record["timings"]
    assert log_path.exists()
    line = log_path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["streams"] == 2
    assert payload["over_budget"] is False
