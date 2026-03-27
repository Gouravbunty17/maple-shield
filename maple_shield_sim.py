"""
maple_shield_sim.py — Synthetic Scenario Simulator for Maple Shield

Generates a realistic multi-drone scenario with no camera required.
Runs the full Maple Shield pipeline:
  Synthetic tracks → Threat scorer → MQTT → CoT UDP → Overlay window → JSONL log

Scenarios:
  recon      Single drone conducting perimeter reconnaissance
  swarm      Three drones approaching from different vectors
  incursion  Drone breaches protected zone, escalates to CRITICAL
  standoff   Two drones loitering at distance then one approaches

Usage:
    python maple_shield_sim.py                         # default: incursion
    python maple_shield_sim.py --scenario swarm        # swarm attack demo
    python maple_shield_sim.py --scenario recon        # single recon drone
    python maple_shield_sim.py --no-display            # headless (log + MQTT only)
    python maple_shield_sim.py --scenario incursion --loops 3

The simulator produces:
  - Live OpenCV display (same HUD as maple_shield_mvp.py)
  - JSONL run log in runs/
  - MQTT alerts to localhost:1883
  - CoT XML events to 239.2.3.1:6969 (ATAK multicast)
"""

from __future__ import annotations

import argparse
import json
import math
import time
import random
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np

from threat_scorer import (
    ScorerConfig, ThreatLevel, score_detection, ScoredDetection
)
from maple_shield_mqtt import AlertPublisher
from maple_shield_cot import CotPublisher, CotConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FW, FH = 1280, 720
FPS = 25.0
BACKGROUND_COLOR = (18, 22, 30)    # Dark blue-grey (radar-style)

THREAT_COLORS = {
    ThreatLevel.CLEAR:    (0,   200, 0),
    ThreatLevel.LOW:      (0,   200, 200),
    ThreatLevel.MEDIUM:   (0,   165, 255),
    ThreatLevel.HIGH:     (0,   0,   255),
    ThreatLevel.CRITICAL: (50,  0,   200),
}


# ---------------------------------------------------------------------------
# Synthetic track
# ---------------------------------------------------------------------------

@dataclass
class SyntheticTrack:
    """A simulated drone / object moving across the scene."""
    track_id: int
    label:    str        # "drone", "bird", "airplane"
    conf:     float      # detection confidence (may vary slightly)

    # Position and movement
    x: float             # current centre x (float pixels)
    y: float
    vx: float            # velocity pixels/frame
    vy: float
    ax: float = 0.0      # acceleration (for curved paths)
    ay: float = 0.0

    # Rendering
    box_w: int = 50
    box_h: int = 30

    # State
    alive: bool = True
    age:   int = 0
    prev_x: Optional[float] = None
    prev_y: Optional[float] = None

    def step(self):
        self.prev_x = self.x
        self.prev_y = self.y
        self.vx += self.ax
        self.vy += self.ay
        self.x  += self.vx
        self.y  += self.vy
        self.age += 1
        # Kill track if off-screen by >200px
        if self.x < -200 or self.x > FW + 200 or self.y < -200 or self.y > FH + 200:
            self.alive = False

    @property
    def box(self) -> list[int]:
        hw, hh = self.box_w // 2, self.box_h // 2
        return [int(self.x - hw), int(self.y - hh),
                int(self.x + hw), int(self.y + hh)]

    @property
    def center(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def prev_center(self) -> Optional[tuple[float, float]]:
        if self.prev_x is None:
            return None
        return (self.prev_x, self.prev_y)

    @property
    def velocity_px(self) -> float:
        if self.prev_center is None:
            return 0.0
        return math.hypot(self.x - self.prev_x, self.y - self.prev_y)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

def scenario_incursion(loop: int = 0) -> list[SyntheticTrack]:
    """
    Sentinel Shield showcase scenario:
    1 drone enters top-right, approaches centre zone, escalates to CRITICAL.
    A bird crosses the bottom of frame (false-positive test).
    """
    seed = loop * 137
    rng  = random.Random(seed)
    return [
        SyntheticTrack(
            track_id=1, label="drone", conf=rng.uniform(0.82, 0.94),
            x=FW - 50, y=80,
            vx=rng.uniform(-4.5, -3.5),
            vy=rng.uniform(2.5, 3.5),
            box_w=52, box_h=32,
        ),
        SyntheticTrack(
            track_id=2, label="bird", conf=rng.uniform(0.55, 0.68),
            x=0, y=FH - 120,
            vx=rng.uniform(5.0, 7.0),
            vy=rng.uniform(-1.0, 0.5),
            box_w=28, box_h=16,
        ),
    ]


def scenario_swarm(loop: int = 0) -> list[SyntheticTrack]:
    """
    Three drones converging on protected zone from different vectors.
    """
    rng = random.Random(loop * 97 + 7)
    cx, cy = FW / 2, FH / 2
    tracks = []
    spawn_points = [
        (50, 50, 4.0, 3.5),          # top-left
        (FW - 50, 50, -4.0, 3.5),    # top-right
        (FW // 2, 20, 0.5, 4.5),     # top-centre
    ]
    for i, (sx, sy, base_vx, base_vy) in enumerate(spawn_points):
        tracks.append(SyntheticTrack(
            track_id=10 + i,
            label="drone",
            conf=rng.uniform(0.78, 0.92),
            x=float(sx), y=float(sy),
            vx=base_vx + rng.uniform(-0.5, 0.5),
            vy=base_vy + rng.uniform(-0.5, 0.5),
            box_w=46, box_h=28,
        ))
    return tracks


def scenario_recon(loop: int = 0) -> list[SyntheticTrack]:
    """
    Single drone flies a loose orbit around the protected zone.
    Demonstrates persistent tracking and threat escalation logic.
    """
    rng = random.Random(loop * 53)
    return [
        SyntheticTrack(
            track_id=20,
            label="drone",
            conf=rng.uniform(0.75, 0.90),
            x=float(FW - 80),
            y=float(FH // 2),
            vx=-3.0,
            vy=1.0,
            ay=0.04,   # gentle curve
            box_w=50,
            box_h=30,
        ),
        SyntheticTrack(
            track_id=21,
            label="bird",
            conf=rng.uniform(0.52, 0.65),
            x=50.0,
            y=rng.uniform(100, 300),
            vx=rng.uniform(6.0, 9.0),
            vy=rng.uniform(-1.0, 1.0),
            box_w=26,
            box_h=15,
        ),
    ]


def scenario_standoff(loop: int = 0) -> list[SyntheticTrack]:
    """
    Two drones loiter at range then one makes a run at the zone.
    Tests hovering detection and late escalation.
    """
    rng = random.Random(loop * 211)
    return [
        SyntheticTrack(
            track_id=30, label="drone",
            conf=rng.uniform(0.80, 0.92),
            x=200.0, y=150.0,
            vx=0.3, vy=0.1,      # slow loiter
            ax=rng.choice([-0.02, 0.02]),
            ay=rng.choice([-0.01, 0.01]),
            box_w=44, box_h=27,
        ),
        SyntheticTrack(
            track_id=31, label="drone",
            conf=rng.uniform(0.77, 0.89),
            x=900.0, y=200.0,
            vx=-0.2, vy=0.15,
            ax=rng.choice([-0.015, 0.015]),
            ay=rng.choice([-0.015, 0.015]),
            box_w=44, box_h=27,
        ),
    ]


SCENARIOS = {
    "incursion": scenario_incursion,
    "swarm":     scenario_swarm,
    "recon":     scenario_recon,
    "standoff":  scenario_standoff,
}


# ---------------------------------------------------------------------------
# Overlay rendering (mirrors maple_shield_mvp.py style)
# ---------------------------------------------------------------------------

def draw_background(frame: np.ndarray) -> np.ndarray:
    """Draw a subtle grid on the background for radar effect."""
    for gx in range(0, FW, 80):
        cv2.line(frame, (gx, 0), (gx, FH), (28, 34, 44), 1)
    for gy in range(0, FH, 80):
        cv2.line(frame, (0, gy), (FW, gy), (28, 34, 44), 1)
    return frame


def draw_zone(frame: np.ndarray, scorer_cfg: ScorerConfig) -> np.ndarray:
    cx = scorer_cfg.frame_w // 2
    cy = scorer_cfg.frame_h // 2
    r  = int(scorer_cfg.zone_radius_frac * min(FW, FH))
    cv2.circle(frame, (cx, cy), r, (60, 60, 220), 1)
    cv2.putText(frame, "PROTECTED ZONE", (cx - r + 6, cy - r + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 220), 1)
    return frame


def draw_track(frame: np.ndarray, sd: ScoredDetection) -> np.ndarray:
    x1, y1, x2, y2 = sd.box
    color = THREAT_COLORS[sd.threat_level]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag = f"ID{sd.track_id} {sd.label} {sd.conf:.2f} [{sd.threat_level.label()}]"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    ty = max(0, y1 - 6)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, tag, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return frame


def draw_hud(frame: np.ndarray, frame_id: int, fps: float,
             scenario: str, max_threat: ThreatLevel,
             mqtt_ok: bool, cot_ok: bool, n_tracks: int) -> np.ndarray:
    hud = (f"MAPLE SHIELD SIM  |  Scenario: {scenario.upper()}"
           f"  |  Frame {frame_id:4d}  |  FPS {fps:.1f}"
           f"  |  Tracks {n_tracks}  |  MAX: {max_threat.label()}"
           f"  |  MQTT: {'ON' if mqtt_ok else 'OFF'}"
           f"  |  CoT: {'ON' if cot_ok else 'OFF'}")
    cv2.rectangle(frame, (0, 0), (FW, 36), (20, 20, 28), -1)
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (200, 220, 220), 1)

    bar_lvls = list(ThreatLevel)
    bx = FW - 140
    for i, lvl in enumerate(bar_lvls):
        c = THREAT_COLORS[lvl] if lvl <= max_threat else (50, 50, 55)
        cv2.rectangle(frame, (bx + i * 26, 4), (bx + i * 26 + 22, 32), c, -1)

    # Watermark
    cv2.putText(frame, "MAPLE SILICON INC. — MAPLE SHIELD",
                (10, FH - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (60, 70, 90), 1)
    return frame


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    scenario_name: str,
    loops: int,
    display: bool,
    runs_dir: Path,
):
    scenario_fn = SCENARIOS[scenario_name]

    run_dir = runs_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    scorer_cfg = ScorerConfig(frame_w=FW, frame_h=FH)

    mqtt_pub = AlertPublisher(
        session_id=run_dir.name,
        frame_w=FW, frame_h=FH,
        min_alert_threat="MEDIUM",
    )
    mqtt_ok = mqtt_pub.connect()

    cot_pub = CotPublisher(CotConfig(frame_w=FW, frame_h=FH))
    cot_pub.start()

    log_path = run_dir / "detections.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")

    writer_path = run_dir / "overlay.mp4"
    writer = None
    if display or True:   # always write video for demo evidence
        writer = cv2.VideoWriter(
            str(writer_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS, (FW, FH)
        )

    (run_dir / "meta.json").write_text(json.dumps({
        "scenario": scenario_name,
        "loops": loops,
        "frame_w": FW,
        "frame_h": FH,
        "started_ts": time.time(),
        "session_id": run_dir.name,
    }, indent=2))

    print(f"\n=== Maple Shield Scenario Simulator ===")
    print(f"  Scenario  : {scenario_name}")
    print(f"  Loops     : {loops}")
    print(f"  MQTT      : {'connected' if mqtt_ok else 'offline'}")
    print(f"  CoT UDP   : {cot_pub.config.udp_targets}")
    print(f"  Output    : {run_dir}")
    print(f"\n  Press Q to quit | Press SPACE to skip to next loop\n")

    frame_id  = 0
    t0        = time.time()
    loop_num  = 0

    while loop_num < loops:
        tracks = scenario_fn(loop_num)

        while any(t.alive for t in tracks):
            ts = time.time()
            frame_id += 1

            # Build frame
            frame = np.full((FH, FW, 3), BACKGROUND_COLOR, dtype=np.uint8)
            draw_background(frame)
            draw_zone(frame, scorer_cfg)

            # Step and score all alive tracks
            det_list = []
            for t in tracks:
                if not t.alive:
                    continue
                t.step()
                if not t.alive:
                    continue
                det_list.append(t)

            scored: list[ScoredDetection] = []
            for t in det_list:
                # Add slight confidence jitter for realism
                jitter = random.uniform(-0.02, 0.02)
                sd = score_detection(
                    track_id      = t.track_id,
                    label         = t.label,
                    conf          = min(0.99, max(0.30, t.conf + jitter)),
                    box           = t.box,
                    track_confirmed = t.age >= 5,
                    prev_center   = t.prev_center,
                    config        = scorer_cfg,
                )
                # Stash prev_center on sd for CoT module
                sd._prev_center = t.prev_center  # type: ignore[attr-defined]
                scored.append(sd)

            max_threat = max((s.threat_level for s in scored),
                            default=ThreatLevel.CLEAR)
            fps = frame_id / max(1e-6, ts - t0)

            # Render
            for sd in scored:
                if sd.object_type != "ignore":
                    draw_track(frame, sd)

            draw_hud(frame, frame_id, fps, scenario_name, max_threat,
                    mqtt_ok, True, len(scored))

            if writer:
                writer.write(frame)

            if display:
                cv2.imshow("MAPLE SHIELD — Scenario Simulator", frame)
                key = cv2.waitKey(int(1000 / FPS)) & 0xFF
                if key == ord("q"):
                    goto_end = True
                    break
                if key == ord(" "):
                    break
            else:
                # Headless: real-time pacing
                elapsed = time.time() - ts
                sleep_t = max(0.0, 1.0 / FPS - elapsed)
                time.sleep(sleep_t)

            # MQTT
            mqtt_pub.on_frame(
                frame_id, scored, max_threat.label(), fps, "DENSE"
            )

            # CoT
            cot_pub.on_frame(frame_id, scored, run_dir.name, fps)

            # JSONL
            log_f.write(json.dumps({
                "ts": ts, "frame": frame_id, "fps": round(fps, 1),
                "max_threat": max_threat.label(),
                "kernel_mode": "DENSE",
                "detections": [{
                    "track_id":   s.track_id,
                    "label":      s.label,
                    "object_type": s.object_type,
                    "conf":       s.conf,
                    "box":        s.box,
                    "confirmed":  s.track_confirmed,
                    "threat":     s.threat_level.label(),
                    "score":      s.threat_score,
                    "in_zone":    s.in_zone,
                    "velocity_px": s.velocity_px,
                } for s in scored],
            }) + "\n")
        else:
            loop_num += 1
            continue
        # Q was pressed
        break

    # Cleanup
    log_f.close()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    mqtt_pub.disconnect()
    cot_pub.stop()

    print(f"\nSimulation complete")
    print(f"  Frames    : {frame_id}")
    print(f"  MQTT alerts: {mqtt_pub.alerts_published}")
    print(f"  CoT events : {cot_pub.events_sent}")
    print(f"  Output dir : {run_dir}")
    print(f"  Video      : {writer_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maple Shield Scenario Simulator")
    parser.add_argument("--scenario",   default="incursion",
                        choices=list(SCENARIOS.keys()),
                        help="Scenario to run (default: incursion)")
    parser.add_argument("--loops",      default=1, type=int,
                        help="Number of times to loop the scenario (default: 1)")
    parser.add_argument("--no-display", action="store_true",
                        help="Headless mode — no OpenCV window")
    parser.add_argument("--runs-dir",   default="runs",
                        help="Output directory for logs and video")
    args = parser.parse_args()

    run_simulation(
        scenario_name = args.scenario,
        loops         = args.loops,
        display       = not args.no_display,
        runs_dir      = Path(args.runs_dir),
    )
