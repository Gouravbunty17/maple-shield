"""Cairn-Edge C2 emitter helpers.

This module intentionally implements reporting/interoperability only. It does
not expose weapon-control, engagement, or effector tasking APIs.
"""
from __future__ import annotations

import json
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from xml.sax.saxutils import escape


def _utc(dt: Optional[datetime] = None) -> str:
    return (dt or datetime.now(timezone.utc)).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CotPoint:
    lat: float
    lon: float
    hae_m: float = 0.0
    ce_m: float = 9999999.0
    le_m: float = 9999999.0


class CotEmitter:
    def __init__(self, host: str, port: int, proto: str = "udp") -> None:
        self.host = host
        self.port = int(port)
        self.proto = proto.lower()
        if self.proto not in {"udp", "tcp"}:
            raise ValueError("CoT proto must be 'udp' or 'tcp'")

    @staticmethod
    def build_event_xml(
        *,
        uid: str,
        callsign: str,
        point: CotPoint,
        threat_level: str,
        remarks: str = "",
        stale_seconds: int = 30,
    ) -> str:
        now = datetime.now(timezone.utc)
        stale = now + timedelta(seconds=stale_seconds)
        cot_type = "a-f-A-M-F-Q"  # Generic air track reporting symbol placeholder.
        return (
            f'<event version="2.0" uid="{escape(uid)}" type="{cot_type}" how="m-g" '
            f'time="{_utc(now)}" start="{_utc(now)}" stale="{_utc(stale)}">'
            f'<point lat="{point.lat:.7f}" lon="{point.lon:.7f}" hae="{point.hae_m:.1f}" ce="{point.ce_m:.1f}" le="{point.le_m:.1f}" />'
            f'<detail><contact callsign="{escape(callsign)}" />'
            f'<remarks>{escape(remarks)}</remarks>'
            f'<cairn threat_level="{escape(threat_level)}" reporting_only="true" /></detail>'
            f'</event>'
        )

    def send_xml(self, xml: str) -> None:
        data = xml.encode("utf-8")
        if self.proto == "udp":
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.sendto(data, (self.host, self.port))
        else:
            with socket.create_connection((self.host, self.port), timeout=3.0) as sock:
                sock.sendall(data)


class MqttPayloadBuilder:
    @staticmethod
    def build_track_update(*, node_id: str, track: Dict[str, Any], risk: Dict[str, Any]) -> str:
        payload = {
            "schema": "cairn.edge.track_update.v1",
            "node_id": node_id,
            "message_id": str(uuid.uuid4()),
            "ts": _utc(),
            "track": track,
            "risk": risk,
            "reporting_only": True,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


class MisbKlvPlaceholder:
    """Placeholder for future MISB ST 0601 KLV generation.

    Full MISB/STANAG implementation requires a validated metadata encoder and
    video mux path. On Orin Nano, H.265/H.264 encode is software unless paired
    with an external encode appliance, so this is intentionally a planning stub.
    """

    def build_metadata_packet(self, metadata: Dict[str, Any]) -> bytes:
        raise NotImplementedError("MISB ST 0601 KLV muxing requires the production encoder path.")
