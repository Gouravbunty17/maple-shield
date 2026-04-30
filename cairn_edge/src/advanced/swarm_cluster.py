"""CPU DBSCAN swarm clustering for Cairn-Edge.

Design target: <5 ms/frame for <=50 tracks on Jetson Orin Nano CPU.
The implementation avoids sklearn so the edge image stays lightweight.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .models import HealthStatus, Track

EARTH_RADIUS_M = 6_371_000.0
SWARM_COT_TYPE = "a-h-A-M-F-Q"  # MIL-STD-2525D hostile air uncrewed reporting symbol.


@dataclass(frozen=True)
class SwarmCluster:
    cluster_id: str
    centroid_lat: float
    centroid_lon: float
    track_ids: List[str]
    radius_m: float
    track_count: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "centroid_lat": self.centroid_lat,
            "centroid_lon": self.centroid_lon,
            "track_ids": list(self.track_ids),
            "radius_m": self.radius_m,
            "track_count": self.track_count,
        }


def _meters_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Equirectangular projection to local meters around lat0/lon0."""
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)
    x = (lon_rad - lon0_rad) * math.cos((lat_rad + lat0_rad) * 0.5) * EARTH_RADIUS_M
    y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
    return x, y


def _centroid(tracks: Sequence[Track]) -> Tuple[float, float]:
    return (
        sum(t.lat for t in tracks) / max(1, len(tracks)),
        sum(t.lon for t in tracks) / max(1, len(tracks)),
    )


def _radius_m(tracks: Sequence[Track], centroid_lat: float, centroid_lon: float) -> float:
    if not tracks:
        return 0.0
    return max(math.hypot(*_meters_xy(t.lat, t.lon, centroid_lat, centroid_lon)) for t in tracks)


class _TinyDBSCAN:
    """Dependency-free DBSCAN suitable for <=50 points."""

    NOISE = -1
    UNVISITED = -99

    def __init__(self, eps_m: float, min_samples: int) -> None:
        self.eps_m = float(eps_m)
        self.min_samples = int(min_samples)

    def fit_predict(self, points: Sequence[Tuple[float, float]]) -> List[int]:
        labels = [self.UNVISITED] * len(points)
        cluster_id = 0
        for idx in range(len(points)):
            if labels[idx] != self.UNVISITED:
                continue
            neighbors = self._region_query(points, idx)
            if len(neighbors) < self.min_samples:
                labels[idx] = self.NOISE
                continue
            self._expand(points, labels, idx, neighbors, cluster_id)
            cluster_id += 1
        return labels

    def _region_query(self, points: Sequence[Tuple[float, float]], idx: int) -> List[int]:
        px, py = points[idx]
        eps2 = self.eps_m * self.eps_m
        return [i for i, (x, y) in enumerate(points) if (px - x) ** 2 + (py - y) ** 2 <= eps2]

    def _expand(
        self,
        points: Sequence[Tuple[float, float]],
        labels: List[int],
        idx: int,
        neighbors: List[int],
        cluster_id: int,
    ) -> None:
        labels[idx] = cluster_id
        queue = list(neighbors)
        cursor = 0
        while cursor < len(queue):
            n_idx = queue[cursor]
            cursor += 1
            if labels[n_idx] == self.NOISE:
                labels[n_idx] = cluster_id
            if labels[n_idx] != self.UNVISITED:
                continue
            labels[n_idx] = cluster_id
            n_neighbors = self._region_query(points, n_idx)
            if len(n_neighbors) >= self.min_samples:
                for candidate in n_neighbors:
                    if candidate not in queue:
                        queue.append(candidate)


class SwarmClusterer:
    """Persistent swarm detector using CPU DBSCAN over geolocated tracks."""

    def __init__(
        self,
        eps_m: float = 50.0,
        min_samples: int = 3,
        persistence_frames: int = 5,
        cot_emitter: Optional[object] = None,
    ) -> None:
        self.eps_m = eps_m
        self.min_samples = min_samples
        self.persistence_frames = persistence_frames
        self.cot_emitter = cot_emitter
        self._dbscan = _TinyDBSCAN(eps_m=eps_m, min_samples=min_samples)
        self._persistence: Dict[str, int] = {}
        self._already_emitted: set[str] = set()
        self._last_health = time.time()
        self._last_reason: Optional[str] = None

    def update(self, tracks: Sequence[Track]) -> Dict[str, Dict[str, object]]:
        """Return persistent clusters only.

        A cluster must be present for at least `persistence_frames` consecutive
        calls before being returned/emitted.
        """
        if len(tracks) < self.min_samples:
            self._persistence.clear()
            return {}

        lat0, lon0 = _centroid(tracks)
        points = [_meters_xy(t.lat, t.lon, lat0, lon0) for t in tracks]
        labels = self._dbscan.fit_predict(points)

        by_label: Dict[int, List[Track]] = {}
        for label, track in zip(labels, tracks):
            if label >= 0:
                by_label.setdefault(label, []).append(track)

        current_keys: set[str] = set()
        output: Dict[str, Dict[str, object]] = {}

        for _label, members in by_label.items():
            if len(members) < self.min_samples:
                continue
            key = "swarm-" + "-".join(sorted(t.track_id for t in members)[:8])
            current_keys.add(key)
            self._persistence[key] = self._persistence.get(key, 0) + 1
            if self._persistence[key] < self.persistence_frames:
                continue
            c_lat, c_lon = _centroid(members)
            cluster = SwarmCluster(
                cluster_id=key,
                centroid_lat=c_lat,
                centroid_lon=c_lon,
                track_ids=[t.track_id for t in members],
                radius_m=_radius_m(members, c_lat, c_lon),
                track_count=len(members),
            )
            output[key] = cluster.as_dict()
            if key not in self._already_emitted:
                self.emit_cot(cluster)
                self._already_emitted.add(key)

        for key in list(self._persistence.keys()):
            if key not in current_keys:
                self._persistence.pop(key, None)
                self._already_emitted.discard(key)

        self._last_health = time.time()
        return output

    def emit_cot(self, cluster: SwarmCluster) -> None:
        """Emit a reporting-only CoT swarm event, if a CotEmitter is configured."""
        if self.cot_emitter is None:
            return
        remarks = f"CAIRN swarm cluster: size={cluster.track_count}, radius_m={cluster.radius_m:.1f}"
        try:
            if hasattr(self.cot_emitter, "emit"):
                self.cot_emitter.emit(
                    type_code=SWARM_COT_TYPE,
                    lat=cluster.centroid_lat,
                    lon=cluster.centroid_lon,
                    remarks=remarks,
                    callsign=cluster.cluster_id,
                )
                return
            if hasattr(self.cot_emitter, "build_event_xml") and hasattr(self.cot_emitter, "send_xml"):
                xml = self._build_cot_xml(cluster, remarks)
                self.cot_emitter.send_xml(xml)
        except Exception as exc:  # reporting failure must not stop detection
            self._last_reason = f"CoT emission failed: {exc}"

    @staticmethod
    def _build_cot_xml(cluster: SwarmCluster, remarks: str) -> str:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        stale = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 30))
        return (
            f'<event version="2.0" uid="{cluster.cluster_id}" type="{SWARM_COT_TYPE}" how="m-g" '
            f'time="{now}" start="{now}" stale="{stale}">'
            f'<point lat="{cluster.centroid_lat:.7f}" lon="{cluster.centroid_lon:.7f}" hae="0" ce="9999999" le="9999999" />'
            f'<detail><remarks>{remarks}</remarks><cairn reporting_only="true" /></detail></event>'
        )

    def health(self) -> HealthStatus:
        status = "degraded" if self._last_reason else "ok"
        return HealthStatus(
            module_name="swarm_cluster",
            status=status,
            last_heartbeat=self._last_health,
            degraded_reason=self._last_reason,
        )
