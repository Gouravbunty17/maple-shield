"""Detector contract: only emits 'drone' class, deterministic over frame index."""

from edge_agent.detector import CairnMockDetector


def test_detector_only_emits_drone_class():
    d = CairnMockDetector()
    for i in range(60):
        for det in d.detect(i, 640, 360):
            assert det.cls == "drone"


def test_detector_is_deterministic():
    d1 = CairnMockDetector()
    d2 = CairnMockDetector()
    for i in range(60):
        a = d1.detect(i, 640, 360)
        b = d2.detect(i, 640, 360)
        assert [(x.confidence, x.bbox) for x in a] == [(x.confidence, x.bbox) for x in b]


def test_detector_drops_some_frames():
    d = CairnMockDetector()
    misses = sum(1 for i in range(100) if not d.detect(i, 640, 360))
    assert misses > 0
    assert misses < 100  # detection still happens most of the time
