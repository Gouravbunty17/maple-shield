import time

from cairn_edge.src.advanced.models import Track
from cairn_edge.src.advanced.swarm_cluster import SwarmClusterer


def make_track(i, lat, lon):
    return Track(track_id=f"t{i}", lat=lat, lon=lon, confidence=0.9, class_id="uas_group_1", kinematic_risk=70, timestamp=time.time())


def test_dbscan_synthetic_points_forms_cluster_after_persistence():
    clusterer = SwarmClusterer(eps_m=50, min_samples=3, persistence_frames=5)
    tracks = [
        make_track(1, 43.680000, -79.625000),
        make_track(2, 43.680050, -79.625050),
        make_track(3, 43.680080, -79.625010),
    ]
    out = {}
    for _ in range(4):
        out = clusterer.update(tracks)
        assert out == {}
    out = clusterer.update(tracks)
    assert len(out) == 1
    cluster = next(iter(out.values()))
    assert cluster["track_count"] == 3
    assert set(cluster["track_ids"]) == {"t1", "t2", "t3"}
    assert cluster["radius_m"] < 50


def test_cluster_resets_when_tracks_disappear():
    clusterer = SwarmClusterer(eps_m=50, min_samples=3, persistence_frames=2)
    tracks = [make_track(1, 43.68, -79.625), make_track(2, 43.68005, -79.62505), make_track(3, 43.68008, -79.62501)]
    assert clusterer.update(tracks) == {}
    assert clusterer.update(tracks)
    assert clusterer.update([]) == {}
    assert clusterer.update(tracks) == {}
