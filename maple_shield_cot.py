"""
maple_shield_cot.py — Cursor on Target (CoT) output for Maple Shield

Generates NATO-compatible CoT 2.0 XML events from Maple Shield detections
and broadcasts them over UDP to ATAK / WinTAK / TAK Server.

CoT Event Types:
  a-h-A-M-F-Q  Hostile  Air Military Fixed-wing UAS  (CRITICAL / HIGH drone)
  a-s-A-M-F-Q  Suspect  Air Military Fixed-wing UAS  (MEDIUM drone)
  a-u-A        Unknown  Air                           (LOW / unclassified)
  a-n-A-X      Neutral  Air Non-military              (bird)

Reference: MIL-STD-6017 / TAK CoT schema v2.0

Geo projection (no GPS):
  Frame centre  →  reference_lat / reference_lon
  Pixel offset  →  metres via metres_per_pixel scale
  Altitude      →  estimated from object pixel size

Usage:
    from maple_shield_cot import CotPublisher, CotConfig
    cot = CotPublisher()
    cot.start()
    cot.on_frame(frame_id, scored_detections, session_id)
    cot.stop()

Standalone broadcast test:
    python maple_shield_cot.py
"""

from __future__ import annotations

import math
import socket
import threading
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CotConfig:
    # Reference geographic position (frame centre maps to this point)
    # Default: CFB Suffield area (Alberta, Canada) — representative test site
    reference_lat:  float = 50.2650
    reference_lon:  float = -110.7300
    reference_alt:  float = 750.0          # metres above sea level (MSL)

    # Camera geometry
    metres_per_pixel: float = 0.8          # at ~100 m altitude, ~90° FOV 1280-wide
    frame_w: int = 1280
    frame_h: int = 720

    # UDP broadcast targets
    # TAK multicast (ATAK default): 239.2.3.1:6969
    # WinTAK / TAK Server unicast:  127.0.0.1:4242
    udp_targets: list[tuple[str, int]] = field(
        default_factory=lambda: [("239.2.3.1", 6969)]
    )
    multicast_ttl: int = 1                 # keep on LAN

    # Stale time (seconds) — how long the track remains on map after last update
    stale_seconds: float = 30.0

    # Min threat level to broadcast CoT (CLEAR / LOW / MEDIUM / HIGH / CRITICAL)
    min_cot_threat: str = "LOW"

    # Log CoT XML to file for audit trail
    log_to_file: bool = True
    log_dir: str = "runs"


# ---------------------------------------------------------------------------
# CoT type mapping
# ---------------------------------------------------------------------------

_COT_TYPES = {
    # (object_type, threat_level) → CoT type
    ("drone",   "CRITICAL"): "a-h-A-M-F-Q",   # Hostile Air UAS
    ("drone",   "HIGH"):     "a-h-A-M-F-Q",
    ("drone",   "MEDIUM"):   "a-s-A-M-F-Q",   # Suspect Air UAS
    ("drone",   "LOW"):      "a-u-A-M-F-Q",   # Unknown Air UAS
    ("drone",   "CLEAR"):    "a-u-A-M-F-Q",
    ("bird",    "HIGH"):     "a-u-A",
    ("bird",    "MEDIUM"):   "a-n-A-X",        # Neutral Air Non-mil
    ("bird",    "LOW"):      "a-n-A-X",
    ("bird",    "CLEAR"):    "a-n-A-X",
    ("unknown", "CRITICAL"): "a-h-A",
    ("unknown", "HIGH"):     "a-s-A",
    ("unknown", "MEDIUM"):   "a-u-A",
    ("unknown", "LOW"):      "a-u-A",
    ("unknown", "CLEAR"):    "a-u-A",
}

_DEFAULT_COT_TYPE = "a-u-A"

_THREAT_ORDER = ["CLEAR", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _threat_rank(level: str) -> int:
    try:
        return _THREAT_ORDER.index(level.upper())
    except ValueError:
        return 0


def _cot_type(object_type: str, threat_level: str) -> str:
    return _COT_TYPES.get((object_type.lower(), threat_level.upper()),
                          _DEFAULT_COT_TYPE)


# ---------------------------------------------------------------------------
# Geo projection
# ---------------------------------------------------------------------------

_DEG_PER_METRE_LAT = 1.0 / 111_111.0


def pixels_to_latlon(
    px: float, py: float,
    config: CotConfig,
) -> tuple[float, float]:
    """
    Convert frame pixel (px, py) to (lat, lon) using linear projection
    from the configured reference point at the frame centre.

    North is up, East is right (standard camera orientation assumed).
    """
    cx = config.frame_w / 2.0
    cy = config.frame_h / 2.0
    dx_m = (px - cx) * config.metres_per_pixel   # East  (+) / West  (-)
    dy_m = (cy - py) * config.metres_per_pixel   # North (+) / South (-)

    lat_rad = math.radians(config.reference_lat)
    deg_per_metre_lon = 1.0 / (111_111.0 * math.cos(lat_rad) + 1e-9)

    lat = config.reference_lat + dy_m * _DEG_PER_METRE_LAT
    lon = config.reference_lon + dx_m * deg_per_metre_lon
    return round(lat, 6), round(lon, 6)


def velocity_to_course_speed(
    prev_center: Optional[tuple[float, float]],
    curr_center: tuple[float, float],
    velocity_px: float,
    config: CotConfig,
    fps: float = 25.0,
) -> tuple[float, float]:
    """
    Returns (course_degrees, speed_m_s).
    course: 0=North, 90=East, 180=South, 270=West (standard CoT course).
    """
    if prev_center is None or velocity_px < 0.5:
        return 0.0, 0.0

    dx = curr_center[0] - prev_center[0]   # East positive
    dy = curr_center[1] - prev_center[1]   # Down positive (image coords)

    # Image y increases downward; North = negative dy
    course_rad = math.atan2(dx, -dy)
    course_deg = (math.degrees(course_rad) + 360.0) % 360.0

    speed_m_s = velocity_px * config.metres_per_pixel * fps
    return round(course_deg, 1), round(speed_m_s, 2)


# ---------------------------------------------------------------------------
# CoT XML builder
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def build_cot_xml(
    track_id:    int,
    object_type: str,
    label:       str,
    threat_level: str,
    threat_score: float,
    conf:        float,
    in_zone:     bool,
    box:         list[int],
    prev_center: Optional[tuple[float, float]],
    velocity_px: float,
    session_id:  str,
    config:      CotConfig,
    fps:         float = 25.0,
) -> str:
    """Build a CoT 2.0 XML string for one tracked detection."""
    now   = datetime.now(timezone.utc)
    stale = now + timedelta(seconds=config.stale_seconds)
    uid   = f"MAPLE-SHIELD.{session_id}.TRK-{track_id:04d}"
    ctype = _cot_type(object_type, threat_level)

    # Pixel centre → lat/lon
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    lat, lon = pixels_to_latlon(cx, cy, config)

    # Altitude estimate: use box height as rough proxy for range
    # Larger box → lower altitude. Rough heuristic only.
    box_h = max(1, box[3] - box[1])
    estimated_alt_m = config.reference_alt + max(0.0, (50.0 - box_h) * 2.0)

    # Course / speed
    curr_c = (cx, cy)
    course, speed = velocity_to_course_speed(prev_center, curr_c,
                                             velocity_px, config, fps)

    # Human-readable callsign
    callsign = f"TRK-{track_id:03d}"
    remarks_text = (
        f"Maple Shield | {label.upper()} | Threat: {threat_level} "
        f"| Score: {threat_score:.2f} | Conf: {conf:.2f}"
        f"{' | IN ZONE' if in_zone else ''}"
        f" | Session: {session_id}"
    )

    xml = (
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
        f"<event version=\"2.0\""
        f" uid=\"{xml_escape(uid)}\""
        f" type=\"{ctype}\""
        f" how=\"m-g\""
        f" time=\"{_iso(now)}\""
        f" start=\"{_iso(now)}\""
        f" stale=\"{_iso(stale)}\">"
        f"<point lat=\"{lat}\" lon=\"{lon}\""
        f" hae=\"{estimated_alt_m:.1f}\""
        f" ce=\"500.0\" le=\"500.0\"/>"
        f"<detail>"
        f"<track speed=\"{speed}\" course=\"{course}\"/>"
        f"<uid Droid=\"{xml_escape(callsign)}\"/>"
        f"<contact callsign=\"{xml_escape(callsign)}\"/>"
        f"<remarks>{xml_escape(remarks_text)}</remarks>"
        f"<MapleShield"
        f" track_id=\"{track_id}\""
        f" threat_level=\"{threat_level}\""
        f" threat_score=\"{threat_score}\""
        f" object_type=\"{object_type}\""
        f" conf=\"{conf:.3f}\""
        f" in_zone=\"{'true' if in_zone else 'false'}\""
        f" session_id=\"{xml_escape(session_id)}\"/>"
        f"</detail>"
        f"</event>"
    )
    return xml


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

class CotPublisher:
    """
    Sends CoT XML events over UDP from Maple Shield detections.

    Thread-safe. Falls back to log-only if no network available.
    """

    def __init__(self, config: Optional[CotConfig] = None):
        self.config = config or CotConfig()
        self._sock: Optional[socket.socket] = None
        self._log_file = None
        self._lock = threading.Lock()
        self.events_sent = 0
        self._prev_threats: dict[int, str] = {}
        self._min_rank = _threat_rank(self.config.min_cot_threat)

    # ------------------------------------------------------------------

    def start(self):
        """Open UDP socket and (optionally) log file."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                                       socket.IPPROTO_UDP)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Enable multicast
            self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL,
                                  self.config.multicast_ttl)
            log.info("CoT UDP socket open → targets: %s", self.config.udp_targets)
        except OSError as e:
            log.warning("CoT UDP socket failed: %s — CoT will be logged only", e)
            self._sock = None

    def stop(self):
        if self._sock:
            self._sock.close()
        if self._log_file:
            self._log_file.close()
        log.info("CotPublisher stopped | events_sent=%d", self.events_sent)

    # ------------------------------------------------------------------

    def on_frame(
        self,
        frame_id: int,
        scored_detections: list,
        session_id: str,
        fps: float = 25.0,
    ):
        """
        Called each frame from the detection pipeline.
        Publishes CoT for any detection at or above min_cot_threat.
        """
        for sd in scored_detections:
            if sd.object_type == "ignore":
                continue
            level = sd.threat_level.label()
            if _threat_rank(level) < self._min_rank:
                continue

            xml_str = build_cot_xml(
                track_id     = sd.track_id,
                object_type  = sd.object_type,
                label        = sd.label,
                threat_level = level,
                threat_score = sd.threat_score,
                conf         = sd.conf,
                in_zone      = sd.in_zone,
                box          = sd.box,
                prev_center  = getattr(sd, "_prev_center", None),
                velocity_px  = sd.velocity_px,
                session_id   = session_id,
                config       = self.config,
                fps          = fps,
            )

            with self._lock:
                self._broadcast(xml_str)
                self.events_sent += 1
                log.debug("CoT sent  track=%d  type=%s  threat=%s",
                          sd.track_id, _cot_type(sd.object_type, level), level)

            self._prev_threats[sd.track_id] = level

    # ------------------------------------------------------------------

    def _broadcast(self, xml_str: str):
        payload = xml_str.encode("utf-8")
        for host, port in self.config.udp_targets:
            try:
                if self._sock:
                    self._sock.sendto(payload, (host, port))
            except OSError as e:
                log.debug("CoT UDP send error %s:%d — %s", host, port, e)
        # Always log to stdout at DEBUG level
        log.debug("CoT XML: %s", xml_str[:120])

    # ------------------------------------------------------------------

    def broadcast_raw(self, xml_str: str):
        """Send a pre-built CoT XML string directly (for testing)."""
        with self._lock:
            self._broadcast(xml_str)
            self.events_sent += 1


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = CotConfig(
        reference_lat=50.2650,
        reference_lon=-110.7300,
        udp_targets=[("127.0.0.1", 4242)],  # unicast to local TAK server
    )
    pub = CotPublisher(cfg)
    pub.start()

    print("=== Maple Shield CoT Test ===")
    print(f"  Reference: {cfg.reference_lat}, {cfg.reference_lon}")
    print(f"  Target:    {cfg.udp_targets}")
    print()

    # Simulate a CRITICAL drone track crossing the frame
    from collections import namedtuple
    FakeDet = namedtuple(
        "FakeDet",
        "track_id object_type label threat_level threat_score conf in_zone "
        "box velocity_px _prev_center track_confirmed"
    )

    class FakeThreat:
        def __init__(self, n): self._n = n
        def label(self): return ["CLEAR","LOW","MEDIUM","HIGH","CRITICAL"][self._n]

    for frame_i in range(10):
        cx = 640 - frame_i * 40
        cy = 360 + frame_i * 10
        det = FakeDet(
            track_id=1,
            object_type="drone",
            label="drone",
            threat_level=FakeThreat(min(4, frame_i // 2 + 1)),
            threat_score=min(0.95, 0.30 + frame_i * 0.07),
            conf=0.88,
            in_zone=frame_i >= 6,
            box=[cx-20, cy-10, cx+20, cy+10],
            velocity_px=float(frame_i * 3),
            _prev_center=(cx+40, cy-10) if frame_i > 0 else None,
            track_confirmed=frame_i >= 2,
        )

        xml_str = build_cot_xml(
            track_id     = det.track_id,
            object_type  = det.object_type,
            label        = det.label,
            threat_level = det.threat_level.label(),
            threat_score = det.threat_score,
            conf         = det.conf,
            in_zone      = det.in_zone,
            box          = det.box,
            prev_center  = det._prev_center,
            velocity_px  = det.velocity_px,
            session_id   = "TEST_20260327",
            config       = cfg,
            fps          = 25.0,
        )

        lat, lon = pixels_to_latlon(cx, cy, cfg)
        print(f"Frame {frame_i:2d} | Track 1 | {det.threat_level.label():8s} | "
              f"lat={lat:.4f} lon={lon:.4f} | CoT sent → {cfg.udp_targets[0]}")
        pub.broadcast_raw(xml_str)
        time.sleep(0.2)

    pub.stop()
    print(f"\nTotal CoT events: {pub.events_sent}")
    print("\nCopy sample CoT XML:\n")
    print(xml_str)
