# Maple Shield Demo Recording Script

Target length: 2-3 minutes
Record with: OBS Studio or Windows Game Bar

---

## Before You Start

- [ ] Clean your desktop and close unrelated windows.
- [ ] Open a terminal in the Maple Shield repo root, for example `C:\Users\15879\Downloads\maple-shield\`.
- [ ] Run `run_demo.bat` and wait for both dashboards to open.
- [ ] Have a video file ready, or use the simulator for a no-camera demo.
- [ ] Set screen resolution to 1920x1080.
- [ ] Start screen recording.

---

## Demo Flow

### Part 1: Introduction

Show the `maplesilicon.co/maple-shield` website briefly.

Say:

> This is Maple Shield, an edge AI platform for real-time drone detection and airspace awareness, built by Maple Silicon Inc. in Caledon, Ontario. Let me show you the system running live.

### Part 2: Live Detection

Switch to terminal and run one of the following.

With your own video:

```bash
python maple_shield_mvp.py --source your_video.mp4
```

Synthetic scenario, no camera required:

```bash
python maple_shield_sim.py --scenario incursion
```

Multi-drone scenario:

```bash
python maple_shield_sim.py --scenario swarm --loops 2
```

Point out:

- Bounding boxes around detected objects.
- Track IDs staying consistent across frames.
- Threat level labels changing from CLEAR to LOW, MEDIUM, HIGH, or CRITICAL.
- Threat bar changing as the risk score changes.
- FPS counter showing real-time performance.
- CoT indicator confirming reporting output.
- JSONL log file being written.

Say:

> The system detects objects in real time using a YOLOv8 model adapted for drone detection. Each detection gets a persistent track ID and a risk score. Alerts can be exported over MQTT and Cursor on Target for authorized command-and-control environments.

### Part 3: C2 Simulator

Switch to browser:

```text
http://localhost:5001
```

Point out:

- Live alert feed updating in real time.
- Threat level badges.
- MQTT stream.
- Timestamp, track ID, and threat level on each alert.

Say:

> Alerts stream in real time for operator review. This gives authorized users a structured feed of what was detected, when it happened, and how the risk score changed.

### Part 4: Replay Dashboard

Switch to browser:

```text
http://localhost:5000
```

Point out:

- Timeline of the detection run.
- Incident replay.
- Track history.
- Evidence log for frame-by-frame review.

Say:

> Every incident is logged and replayable. Operators can review exactly what the system detected, when, and why the risk score changed. This audit trail is critical for accountable airspace monitoring.

### Part 5: Close

Switch back to the website or terminal.

Say:

> Maple Shield runs on compact edge hardware, requires no cloud dependency for core detection, and keeps an audit-ready record of airspace events. Maple Silicon Inc. - Protecting Airspace With Intelligence.

---

## After Recording

- [ ] Export as MP4, 1080p.
- [ ] Upload to YouTube unlisted or Vimeo.
- [ ] Add the link to your IDEaS / Maple Shield application.
- [ ] Add a "Watch Demo" button to `maplesilicon.co/maple-shield`.

---

## Scenario Quick Reference

| Scenario | Command | Best for |
|---|---|---|
| Single drone incursion | `python maple_shield_sim.py --scenario incursion` | General demo with full risk escalation |
| Multi-drone scenario | `python maple_shield_sim.py --scenario swarm` | Multi-target tracking |
| Recon orbit | `python maple_shield_sim.py --scenario recon` | Persistent tracking and loiter detection |
| Standoff plus approach | `python maple_shield_sim.py --scenario standoff` | Risk scoring nuance |
| Loop multiple times | Add `--loops 3` to any command | Longer recording |

---

## Tips for a Clean Recording

| Do | Do Not |
|---|---|
| Speak slowly and clearly | Rush through screens |
| Zoom in on key numbers and visible tracks | Leave unrelated windows visible |
| Pause briefly when switching windows | Use noisy background audio |
| Show terminal output where useful | Over-edit the demo |
| Use the multi-drone scenario for visual density | Use a webcam if you do not have useful footage |
