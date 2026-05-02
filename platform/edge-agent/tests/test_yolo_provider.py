import numpy as np
import pytest

from edge_agent.yolo_provider import YoloDetectionProvider


class _Input:
    name = "images"


class _FakeSession:
    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.float32)
        self.feeds = []

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, input_feed):
        self.feeds.append(input_feed)
        return [self.rows]


def _frame():
    return np.zeros((360, 640, 3), dtype=np.uint8)


def _provider(rows, **kwargs):
    return YoloDetectionProvider(
        _FakeSession(rows),
        class_names=["bird", "drone", "person", "kite"],
        conf_threshold=0.35,
        **kwargs,
    )


def test_drone_class_emits_cairn_detection():
    provider = _provider([[[320, 180, 80, 40, 0.05, 0.91, 0.01, 0.02]]])

    detections = provider(frame_idx=7, frame=_frame(), frame_w=640, frame_h=360)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "drone"
    assert detection.confidence == pytest.approx(0.91)
    assert detection.box == [280, 160, 360, 200]
    assert detection.frame_w == 640
    assert detection.frame_h == 360
    assert detection.source == "yolo_onnx"
    assert detection.raw["frame_idx"] == 7


def test_non_drone_class_dropped_before_cairn_detection():
    provider = _provider(
        [[
            [100, 80, 20, 20, 0.88, 0.05, 0.02, 0.01],
            [220, 120, 30, 30, 0.02, 0.04, 0.93, 0.01],
            [420, 220, 40, 40, 0.03, 0.04, 0.02, 0.94],
        ]]
    )

    assert provider(frame_idx=1, frame=_frame(), frame_w=640, frame_h=360) == []


def test_low_confidence_filtered():
    provider = _provider([[[320, 180, 80, 40, 0.05, 0.34, 0.01, 0.02]]])

    assert provider(frame_idx=1, frame=_frame(), frame_w=640, frame_h=360) == []


def test_track_id_stable_across_consecutive_frames():
    provider = _provider(
        [[[320, 180, 80, 40, 0.05, 0.91, 0.01, 0.02]]],
        track_id_factory=iter([42, 99]).__next__,
    )

    first = provider(frame_idx=1, frame=_frame(), frame_w=640, frame_h=360)
    provider.session.rows = np.asarray([[[323, 182, 80, 40, 0.04, 0.90, 0.01, 0.02]]], dtype=np.float32)
    second = provider(frame_idx=2, frame=_frame(), frame_w=640, frame_h=360)

    assert first[0].track_id == 42
    assert second[0].track_id == 42
    assert second[0].track_confirmed is True
    assert second[0].persistence_frames == 2
    assert second[0].velocity_px > 0
