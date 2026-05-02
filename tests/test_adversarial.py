import numpy as np

from cairn_edge.src.advanced.adversarial_detector import AdversarialPatchDetector
from cairn_edge.src.advanced.models import Detection


class FixedBackend:
    def __init__(self, logits):
        self.logits = np.array(logits, dtype=np.float32)

    def predict_logits(self, roi_rgb_64):
        assert roi_rgb_64.shape == (64, 64, 3)
        return self.logits


def test_process_high_confidence_detection_flags_adversarial():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[20:80, 20:80] = 255
    detection = Detection(bbox=(20, 20, 60, 60), confidence=0.95, class_id="uas")
    detector = AdversarialPatchDetector("missing.engine", backend=FixedBackend([0.0, 3.0]), threshold=0.8)
    out = detector.process_detections(frame, [detection])[0]
    assert out.adversarial is True
    assert out.adversarial_score > 0.8
    assert out.roi_crop is not None


def test_low_confidence_detection_skips_backend():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    detection = Detection(bbox=(20, 20, 60, 60), confidence=0.5, class_id="uas")
    detector = AdversarialPatchDetector("missing.engine", backend=FixedBackend([0.0, 3.0]), threshold=0.8)
    out = detector.process_detections(frame, [detection])[0]
    assert out.adversarial is False
    assert out.adversarial_score == 0.0


def test_missing_engine_degrades_and_skips():
    detector = AdversarialPatchDetector("does-not-exist.engine")
    status = detector.health()
    assert status.status == "degraded"
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    detection = Detection(bbox=(0, 0, 64, 64), confidence=0.95, class_id="uas")
    out = detector.process_detections(frame, [detection])[0]
    assert out.adversarial is False
