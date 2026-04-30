import json
from pathlib import Path

from cairn_edge.src.advanced.geofence_engine import GeofenceEngine
from cairn_edge.src.advanced.models import Track
from cairn_edge.src.advanced.swarm_cluster import SwarmClusterer


def test_20s_replay_cluster_and_zone_events(tmp_path):
    cfg = tmp_path / "zones.yaml"
    cfg.write_text("""
zones:
  - name: Inner Zone
    polygon:
      - [43.6830, -79.6280]
      - [43.6830, -79.6220]
      - [43.6770, -79.6220]
      - [43.6770, -79.6280]
    altitude_min_m: 0
    altitude_max_m: 500
    action: alert
    risk_multiplier: 2.0
""", encoding="utf-8")
    geofence = GeofenceEngine(cfg)
    swarm = SwarmClusterer(eps_m=50, min_samples=3, persistence_frames=5)
    fixture = Path(__file__).parent / "fixtures" / "swarm_tracks_20s.jsonl"

    cluster_frames = 0
    alert_events = 0
    with fixture.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tracks = [Track(**t) for t in row["tracks"]]
            clusters = swarm.update(tracks)
            if clusters:
                cluster_frames += 1
            for track in tracks:
                risk = geofence.evaluate_track(track)
                if risk.action == "alert" and risk.score == 100:
                    alert_events += 1

    assert cluster_frames >= 16
    assert alert_events == 60
