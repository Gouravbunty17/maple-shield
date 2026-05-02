import pytest

from edge_agent.detector import CairnMockDetector, Detection
from edge_agent.main import _build_detector, _detect_frame, _load_class_names, _parser


def _args(*extra):
    return _parser().parse_args(list(extra))


def test_build_detector_defaults_to_mock():
    detector = _build_detector(_args())

    assert isinstance(detector, CairnMockDetector)


def test_cairn_yolo_requires_model_path():
    args = _args("--detector", "cairn-yolo")

    with pytest.raises(ValueError, match="--yolo-model is required"):
        _build_detector(args)


def test_load_class_names_from_comma_list():
    assert _load_class_names("drone,bird, kite") == ["drone", "bird", "kite"]


def test_load_class_names_from_file(tmp_path):
    labels = tmp_path / "labels.txt"
    labels.write_text("drone\nbird\n\nkite\n", encoding="utf-8")

    assert _load_class_names(str(labels)) == ["drone", "bird", "kite"]


def test_build_detector_passes_yolo_options():
    class FakeProvider:
        @classmethod
        def from_onnx(cls, model_path, **kwargs):
            cls.seen = (model_path, kwargs)
            return "provider"

    class FakeCairnDetector:
        def __init__(self, detection_provider, fps):
            self.detection_provider = detection_provider
            self.fps = fps

    args = _args(
        "--detector", "cairn-yolo",
        "--yolo-model", "model.onnx",
        "--yolo-classes", "drone,bird",
        "--conf-threshold", "0.4",
        "--iou-threshold", "0.5",
        "--max-detections", "7",
        "--fps", "12",
    )

    detector = _build_detector(args, provider_cls=FakeProvider, cairn_detector_cls=FakeCairnDetector)

    assert detector.detection_provider == "provider"
    assert detector.fps == 12
    assert FakeProvider.seen == (
        "model.onnx",
        {
            "conf_threshold": 0.4,
            "iou_threshold": 0.5,
            "max_detections": 7,
            "class_names": ["drone", "bird"],
        },
    )


def test_detect_frame_uses_frame_aware_detector():
    class FrameAware:
        def detect_frame(self, frame_idx, frame):
            self.seen = (frame_idx, frame)
            return [Detection(cls="drone", confidence=0.8, bbox=(1, 2, 3, 4))]

    frame = object()
    detector = FrameAware()

    detections = _detect_frame(detector, frame_idx=9, frame=frame, frame_w=640, frame_h=360)

    assert detector.seen == (9, frame)
    assert detections[0].cls == "drone"


def test_detect_frame_falls_back_to_dimension_detector():
    class DimensionOnly:
        def detect(self, frame_idx, frame_w, frame_h):
            self.seen = (frame_idx, frame_w, frame_h)
            return [Detection(cls="drone", confidence=0.7, bbox=(5, 6, 7, 8))]

    detector = DimensionOnly()

    detections = _detect_frame(detector, frame_idx=3, frame=object(), frame_w=640, frame_h=360)

    assert detector.seen == (3, 640, 360)
    assert detections[0].bbox == (5, 6, 7, 8)
