"""
maple_shield_mqtt.py — MQTT Alert Publisher for Maple Shield

Publishes threat escalation alerts to an MQTT broker, simulating
integration with an external C2 (command & control) system.

Install broker (Windows): https://mosquitto.org/download/
Install client:           pip install paho-mqtt

Topics:
  maple_shield/alert        JSON alert when threat escalates to MEDIUM+
  maple_shield/heartbeat    System heartbeat every 5 s
  maple_shield/detections   Full frame snapshot (every frame, optional)

Usage:
    pub = AlertPublisher()
    pub.connect()
    pub.on_frame(frame_id, scored_detections, max_threat, fps)
    pub.disconnect()
"""

from __future__ import annotations

import json
import time
import threading
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional paho-mqtt import — graceful fallback if not installed
# ---------------------------------------------------------------------------
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    log.warning("paho-mqtt not installed — MQTT publishing disabled. "
                "Run: pip install paho-mqtt")


# ---------------------------------------------------------------------------
# Alert data structures
# ---------------------------------------------------------------------------

@dataclass
class ThreatAlert:
    timestamp:    str          # ISO-8601 UTC
    session_id:   str          # run session identifier
    frame_id:     int
    track_id:     int
    prev_threat:  str          # threat level before escalation
    threat_level: str          # current threat level
    threat_score: float        # 0.0–1.0
    object_type:  str          # drone / bird / unknown
    label:        str          # detector class label
    conf:         float        # detection confidence
    box:          list[int]    # [x1, y1, x2, y2] frame pixels
    center_x:     float        # normalised 0–1
    center_y:     float        # normalised 0–1
    velocity_px:  float        # px/frame magnitude
    in_zone:      bool         # inside protected zone
    fps:          float        # system FPS at time of alert

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class SystemHeartbeat:
    timestamp:     str
    session_id:    str
    frame_id:      int
    fps:           float
    kernel_mode:   str          # DENSE / SPARSE
    max_threat:    str
    active_tracks: int
    uptime_s:      float

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


# ---------------------------------------------------------------------------
# MQTT Alert Publisher
# ---------------------------------------------------------------------------

ESCALATION_ORDER = ["CLEAR", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _threat_rank(level: str) -> int:
    try:
        return ESCALATION_ORDER.index(level.upper())
    except ValueError:
        return 0


class AlertPublisher:
    """
    Publishes Maple Shield alerts to an MQTT broker.

    Falls back to no-op mode if paho-mqtt is unavailable or broker
    connection fails — detection pipeline is never blocked.
    """

    TOPIC_ALERT      = "maple_shield/alert"
    TOPIC_HEARTBEAT  = "maple_shield/heartbeat"
    TOPIC_DETECTIONS = "maple_shield/detections"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        session_id: Optional[str] = None,
        frame_w: int = 1280,
        frame_h: int = 720,
        publish_all_frames: bool = False,
        heartbeat_interval_s: float = 5.0,
        min_alert_threat: str = "MEDIUM",
    ):
        self.host = host
        self.port = port
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.publish_all_frames = publish_all_frames
        self.heartbeat_interval_s = heartbeat_interval_s
        self.min_alert_rank = _threat_rank(min_alert_threat)

        self._client = None
        self._connected = False
        self._start_time = time.time()
        self._hb_timer: Optional[threading.Timer] = None

        # Track previous threat per track_id to detect escalations
        self._prev_threats: dict[int, str] = {}

        # Stats
        self.alerts_published = 0
        self.frames_published = 0

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to MQTT broker. Returns True on success."""
        if not MQTT_AVAILABLE:
            log.warning("MQTT disabled (paho-mqtt not installed)")
            return False

        try:
            self._client = mqtt.Client(
                client_id=f"maple_shield_{self.session_id}",
                protocol=mqtt.MQTTv311,
            )
            self._client.on_connect    = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.will_set(
                self.TOPIC_HEARTBEAT,
                json.dumps({"session_id": self.session_id, "status": "offline"}),
                retain=True,
            )
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            time.sleep(0.3)  # brief wait for on_connect callback
            self._schedule_heartbeat()
            return self._connected
        except Exception as exc:
            log.error("MQTT connect failed: %s — running without C2 integration", exc)
            self._client = None
            return False

    def disconnect(self):
        if self._hb_timer:
            self._hb_timer.cancel()
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
        self._connected = False
        log.info("MQTT disconnected | alerts=%d frames=%d",
                 self.alerts_published, self.frames_published)

    # ------------------------------------------------------------------
    # Per-frame entry point (called from MVP main loop)
    # ------------------------------------------------------------------

    def on_frame(
        self,
        frame_id: int,
        scored_detections: list,          # list[ScoredDetection]
        max_threat_label: str,
        fps: float,
        kernel_mode: str = "DENSE",
    ):
        """Process one frame — publish alerts for threat escalations."""
        if not self._connected:
            return

        ts = datetime.now(timezone.utc).isoformat()

        for sd in scored_detections:
            if sd.object_type == "ignore":
                continue

            cur_level = sd.threat_level.label()
            prev_level = self._prev_threats.get(sd.track_id, "CLEAR")

            # Publish alert on escalation above minimum threshold
            is_escalation = _threat_rank(cur_level) > _threat_rank(prev_level)
            above_threshold = _threat_rank(cur_level) >= self.min_alert_rank

            if is_escalation and above_threshold:
                cx = (sd.box[0] + sd.box[2]) / 2.0 / max(1, self.frame_w)
                cy = (sd.box[1] + sd.box[3]) / 2.0 / max(1, self.frame_h)
                alert = ThreatAlert(
                    timestamp    = ts,
                    session_id   = self.session_id,
                    frame_id     = frame_id,
                    track_id     = sd.track_id,
                    prev_threat  = prev_level,
                    threat_level = cur_level,
                    threat_score = sd.threat_score,
                    object_type  = sd.object_type,
                    label        = sd.label,
                    conf         = round(sd.conf, 3),
                    box          = sd.box,
                    center_x     = round(cx, 4),
                    center_y     = round(cy, 4),
                    velocity_px  = sd.velocity_px,
                    in_zone      = sd.in_zone,
                    fps          = round(fps, 1),
                )
                self._publish(self.TOPIC_ALERT, alert.to_json(), qos=1)
                self.alerts_published += 1
                log.info("🚨 ALERT  track=%-3d  %s → %s  score=%.2f  zone=%s",
                         sd.track_id, prev_level, cur_level,
                         sd.threat_score, sd.in_zone)

            self._prev_threats[sd.track_id] = cur_level

        # Optionally publish full frame snapshot
        if self.publish_all_frames:
            payload = json.dumps({
                "ts": ts, "session_id": self.session_id, "frame": frame_id,
                "fps": round(fps, 1), "max_threat": max_threat_label,
                "kernel_mode": kernel_mode,
                "detections": [
                    {
                        "track_id":   sd.track_id,
                        "label":      sd.label,
                        "threat":     sd.threat_level.label(),
                        "score":      sd.threat_score,
                        "conf":       round(sd.conf, 3),
                        "box":        sd.box,
                        "in_zone":    sd.in_zone,
                        "velocity":   sd.velocity_px,
                    }
                    for sd in scored_detections if sd.object_type != "ignore"
                ],
            }, separators=(",", ":"))
            self._publish(self.TOPIC_DETECTIONS, payload, qos=0)
            self.frames_published += 1

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _schedule_heartbeat(self):
        if self._connected:
            self._hb_timer = threading.Timer(
                self.heartbeat_interval_s, self._send_heartbeat
            )
            self._hb_timer.daemon = True
            self._hb_timer.start()

    def _send_heartbeat(self):
        if not self._connected:
            return
        hb = SystemHeartbeat(
            timestamp     = datetime.now(timezone.utc).isoformat(),
            session_id    = self.session_id,
            frame_id      = 0,
            fps           = 0.0,
            kernel_mode   = "UNKNOWN",
            max_threat    = "CLEAR",
            active_tracks = len(self._prev_threats),
            uptime_s      = round(time.time() - self._start_time, 1),
        )
        self._publish(self.TOPIC_HEARTBEAT, hb.to_json(), retain=True)
        self._schedule_heartbeat()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False):
        try:
            self._client.publish(topic, payload, qos=qos, retain=retain)
        except Exception as exc:
            log.debug("MQTT publish error: %s", exc)

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            log.info("MQTT connected → %s:%d", self.host, self.port)
        else:
            log.error("MQTT connect refused rc=%d", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            log.warning("MQTT unexpected disconnect rc=%d", rc)
