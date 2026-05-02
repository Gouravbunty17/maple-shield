import warnings

from cairn_engine import CairnDetection, CairnEngine

from edge_agent.cairn_adapter import (
    CairnSourceDetector,
    CairnVersionWarning,
    check_cairn_version,
)


def _provider(labels):
    def provide(frame_idx: int, frame_w: int, frame_h: int):
        return [
            CairnDetection(
                track_id=i + 1,
                label=label,
                confidence=0.9,
                box=[20 + i * 10, 20, 60 + i * 10, 60],
                frame_w=frame_w,
                frame_h=frame_h,
                track_confirmed=True,
                persistence_frames=12,
            )
            for i, label in enumerate(labels)
        ]

    return provide


def test_cairn_adapter_emits_drone_only():
    detector = CairnSourceDetector(
        engine=CairnEngine(),
        detection_provider=_provider(["drone", "bird", "person", "unknown-object"]),
    )

    detections = detector.detect(frame_idx=1, frame_w=640, frame_h=360)

    assert len(detections) == 1
    assert detections[0].cls == "drone"
    assert detections[0].track_id == "cairn-1"
    assert detections[0].raw["cairn_object_type"] == "drone"
    assert "cairn_threat_level" in detections[0].raw


def test_cairn_adapter_drops_ignore_class():
    detector = CairnSourceDetector(
        engine=CairnEngine(),
        detection_provider=_provider(["person", "car", "truck"]),
    )

    assert detector.detect(frame_idx=1, frame_w=640, frame_h=360) == []


def test_cairn_version_warning_on_minor_drift():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert check_cairn_version("2.1.0") is False

    assert any(isinstance(w.message, CairnVersionWarning) for w in caught)
