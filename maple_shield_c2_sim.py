"""
maple_shield_c2_sim.py — C2 Simulator for Maple Shield

Simulates an external Command & Control system receiving alerts from
Maple Shield via MQTT and displaying them in real time via a web UI.

Usage:
    # Terminal 1: run detection pipeline
    python maple_shield_mvp.py --source video.mp4

    # Terminal 2: run C2 simulator
    python maple_shield_c2_sim.py

    # Browser: http://localhost:5001

Requirements: pip install flask paho-mqtt

Fallback: if no MQTT broker is running, the C2 sim loads the most recent
          JSONL run from runs/ and replays alerts from it.
"""

from __future__ import annotations

import json
import queue
import time
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, render_template, jsonify, request

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Global alert queue (thread-safe, populated by MQTT or replay)
# ---------------------------------------------------------------------------

alert_queue: queue.Queue[dict] = queue.Queue(maxsize=500)
alert_history: list[dict] = []          # kept in memory for /api/history
MAX_HISTORY = 200
_history_lock = threading.Lock()


def _push_alert(alert: dict):
    """Add alert to queue and history."""
    alert.setdefault("received_at", datetime.now(timezone.utc).isoformat())
    alert_queue.put_nowait(alert) if not alert_queue.full() else None
    with _history_lock:
        alert_history.append(alert)
        if len(alert_history) > MAX_HISTORY:
            alert_history.pop(0)


# ---------------------------------------------------------------------------
# MQTT Subscriber (optional)
# ---------------------------------------------------------------------------

MQTT_AVAILABLE = False
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    pass


def start_mqtt_subscriber(host: str = "localhost", port: int = 1883):
    if not MQTT_AVAILABLE:
        log.warning("paho-mqtt not installed — MQTT subscriber disabled")
        return None

    client = mqtt.Client(client_id="maple_shield_c2_sim", protocol=mqtt.MQTTv311)

    def on_connect(c, userdata, flags, rc):
        if rc == 0:
            log.info("C2 Sim connected to MQTT broker %s:%d", host, port)
            c.subscribe("maple_shield/#", qos=1)
        else:
            log.error("MQTT connect refused rc=%d", rc)

    def on_message(c, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            payload["_topic"] = msg.topic
            if msg.topic == "maple_shield/alert":
                _push_alert(payload)
        except Exception as exc:
            log.debug("MQTT message parse error: %s", exc)

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(host, port, keepalive=60)
        t = threading.Thread(target=client.loop_forever, daemon=True)
        t.start()
        return client
    except Exception as exc:
        log.warning("Cannot connect to MQTT broker: %s — using JSONL fallback", exc)
        return None


# ---------------------------------------------------------------------------
# JSONL Fallback Replay
# ---------------------------------------------------------------------------

def replay_jsonl_alerts(runs_dir: str = "runs", delay_s: float = 0.1):
    """
    If no MQTT broker available, find the latest JSONL run and
    replay alerts from it into the alert queue (simulates live feed).
    """
    runs = sorted(Path(runs_dir).glob("*/detections.jsonl"))
    if not runs:
        log.info("No JSONL runs found — C2 sim will wait for live MQTT alerts.")
        return

    jsonl_path = runs[-1]
    log.info("Replaying alerts from: %s", jsonl_path)

    def _replay():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    for det in record.get("detections", []):
                        threat = det.get("threat", "CLEAR")
                        if threat in ("MEDIUM", "HIGH", "CRITICAL"):
                            alert = {
                                "timestamp":    datetime.fromtimestamp(
                                    record["ts"], tz=timezone.utc
                                ).isoformat(),
                                "session_id":   jsonl_path.parent.name,
                                "frame_id":     record.get("frame", 0),
                                "track_id":     det.get("track_id", 0),
                                "prev_threat":  "LOW",
                                "threat_level": threat,
                                "threat_score": det.get("score", 0.0),
                                "object_type":  det.get("object_type", "unknown"),
                                "label":        det.get("label", "unknown"),
                                "conf":         det.get("conf", 0.0),
                                "box":          det.get("box", [0, 0, 0, 0]),
                                "center_x":     0.5,
                                "center_y":     0.5,
                                "velocity_px":  det.get("velocity_px", 0.0),
                                "in_zone":      det.get("in_zone", False),
                                "fps":          record.get("fps", 0.0),
                                "_source":      "replay",
                            }
                            _push_alert(alert)
                except Exception:
                    pass
                time.sleep(delay_s)

    t = threading.Thread(target=_replay, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("c2_sim.html")


@app.route("/api/history")
def api_history():
    """Return full alert history as JSON."""
    with _history_lock:
        return jsonify(list(alert_history))


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events stream — pushes alerts to browser in real time."""
    def event_stream():
        # First, send all existing history
        with _history_lock:
            for alert in alert_history:
                yield f"data: {json.dumps(alert)}\n\n"

        # Then stream new alerts as they arrive
        while True:
            try:
                alert = alert_queue.get(timeout=15)
                yield f"data: {json.dumps(alert)}\n\n"
            except queue.Empty:
                # Keep-alive ping
                yield f": ping\n\n"

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/clear", methods=["POST"])
def api_clear():
    with _history_lock:
        alert_history.clear()
    while not alert_queue.empty():
        try: alert_queue.get_nowait()
        except queue.Empty: break
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maple Shield C2 Simulator")
    parser.add_argument("--host",     default="localhost", help="MQTT broker host")
    parser.add_argument("--port",     default=1883, type=int, help="MQTT broker port")
    parser.add_argument("--web-port", default=5001, type=int, help="Web UI port")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory for JSONL fallback")
    args = parser.parse_args()

    # Try MQTT first, fall back to JSONL replay
    client = start_mqtt_subscriber(args.host, args.port)
    if client is None:
        time.sleep(0.5)
        replay_jsonl_alerts(args.runs_dir)

    print(f"\n🛡️  Maple Shield C2 Simulator")
    print(f"   Web UI  → http://localhost:{args.web_port}")
    print(f"   MQTT    → {'connected' if client else 'disabled (JSONL fallback)'}")
    print(f"   Press Ctrl+C to stop\n")

    app.run(host="0.0.0.0", port=args.web_port, debug=False, threaded=True)
