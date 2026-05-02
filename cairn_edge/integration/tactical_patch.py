"""Integration patch showing how to wire tactical modules into CairnEdgeProcessor.

This file is adapter-style so it can be imported by the current MVP pipeline
without forcing DeepStream/TensorRT dependencies during unit tests.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cairn_edge.src.advanced.geofence_engine import GeofenceEngine
from cairn_edge.src.advanced.mesh_sync import MeshSync
from cairn_edge.src.advanced.models import HealthStatus, Track
from cairn_edge.src.advanced.swarm_cluster import SwarmClusterer

LOGGER = logging.getLogger("cairn_edge.tactical")


@dataclass
class HealthMonitor:
    """Small central health aggregator for tactical modules."""

    statuses: Dict[str, HealthStatus] = field(default_factory=dict)

    def update(self, *modules: object) -> Dict[str, HealthStatus]:
        for module in modules:
            if not hasattr(module, "health"):
                continue
            status = module.health()  # type: ignore[no-any-return]
            self.statuses[status.module_name] = status
            if status.status != "ok":
                LOGGER.warning("Cairn-Edge degraded: %s - %s", status.module_name, status.degraded_reason)
        return dict(self.statuses)


class TacticalPatch:
    """Drop-in tactical stage for the existing CairnEdgeProcessor pipeline."""

    def __init__(
        self,
        swarm_cluster: SwarmClusterer,
        geofence: GeofenceEngine,
        mesh: Optional[MeshSync] = None,
        broadcast_min_interval_s: float = 0.25,
    ) -> None:
        self.swarm_cluster = swarm_cluster
        self.geofence = geofence
        self.mesh = mesh
        self.health_monitor = HealthMonitor()
        self.broadcast_min_interval_s = broadcast_min_interval_s
        self._last_track_broadcast: Dict[str, float] = {}

    def after_tracking_step(self, tracks: List[Track]) -> Dict[str, object]:
        """Call this immediately after local tracking updates.

        Remote tracks are returned for display/fusion only and are not fed back
        into the local tracker.
        """
        clusters = self.swarm_cluster.update(tracks)
        risk_by_track: Dict[str, object] = {}

        now = time.time()
        for track in tracks:
            risk = self.geofence.evaluate_track(track)
            risk_by_track[track.track_id] = risk
            if self.mesh is not None:
                last = self._last_track_broadcast.get(track.track_id, 0.0)
                if now - last >= self.broadcast_min_interval_s:
                    self.mesh.broadcast_track(track)
                    self._last_track_broadcast[track.track_id] = now

        remote_tracks: List[Track] = []
        if self.mesh is not None:
            for _ in range(8):
                self.mesh.receive_once()
            remote_tracks = self.mesh.get_merged_tracks(max_age_s=5.0)

        modules = (self.swarm_cluster, self.geofence, self.mesh) if self.mesh else (self.swarm_cluster, self.geofence)
        health = self.health_monitor.update(*modules)

        return {
            "clusters": clusters,
            "risk_by_track": risk_by_track,
            "remote_tracks_for_display_only": remote_tracks,
            "health": health,
        }


# Example usage inside CairnEdgeProcessor:
#
# tactical = TacticalPatch(
#     swarm_cluster=SwarmClusterer(cot_emitter=cot_emitter),
#     geofence=GeofenceEngine("cairn_edge/src/advanced/geofence_zones.yaml", cot_emitter=cot_emitter),
#     mesh=mesh_sync,
# )
#
# def process_frame(...):
#     local_tracks = tracker.update(detections)
#     tactical_result = tactical.after_tracking_step(local_tracks)
#     display_tracks = local_tracks + tactical_result["remote_tracks_for_display_only"]
