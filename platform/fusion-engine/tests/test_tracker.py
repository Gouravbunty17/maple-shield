"""Tracker unit tests with deterministic synthetic detections."""

from fusion.tracker import Tracker, iou


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    val = iou((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0.1 < val < 0.2


def test_single_track_grows_with_consistent_detection():
    t = Tracker(iou_thresh=0.1, max_misses=2)
    for i in range(10):
        # bbox drifts a bit each frame
        t.update("cam-1", float(i),
                 [{"cls": "drone", "confidence": 0.8,
                   "bbox": [i, i, 30 + i, 30 + i]}])
    assert len(t.tracks) == 1
    trk = list(t.tracks.values())[0]
    assert trk.n_obs == 10
    assert trk.max_conf == 0.8


def test_track_retired_after_misses():
    t = Tracker(iou_thresh=0.1, max_misses=2)
    t.update("cam-1", 0.0, [{"cls": "drone", "confidence": 0.8, "bbox": [0, 0, 10, 10]}])
    assert len(t.tracks) == 1
    # 3 empty frames -> retire
    t.update("cam-1", 1.0, [])
    t.update("cam-1", 2.0, [])
    t.update("cam-1", 3.0, [])
    assert len(t.tracks) == 0


def test_two_simultaneous_drones_get_two_tracks():
    t = Tracker(iou_thresh=0.1, max_misses=2)
    t.update("cam-1", 0.0, [
        {"cls": "drone", "confidence": 0.8, "bbox": [0, 0, 10, 10]},
        {"cls": "drone", "confidence": 0.8, "bbox": [200, 200, 220, 220]},
    ])
    assert len(t.tracks) == 2


def test_non_drone_classes_dropped():
    t = Tracker()
    t.update("cam-1", 0.0, [
        {"cls": "person", "confidence": 0.99, "bbox": [0, 0, 10, 10]},
        {"cls": "vehicle", "confidence": 0.99, "bbox": [50, 50, 60, 60]},
    ])
    # nothing tracked because we only accept 'drone'
    assert len(t.tracks) == 0


def test_velocity_estimate_increases_for_moving_drone():
    t = Tracker(iou_thresh=0.1, max_misses=2, smoothing=0.5)
    for i in range(8):
        t.update("cam-1", float(i),
                 [{"cls": "drone", "confidence": 0.9,
                   "bbox": [10 * i, 0, 10 * i + 20, 20]}])
    trk = list(t.tracks.values())[0]
    vx, vy = trk.velocity
    assert vx > 0  # moving in +x
