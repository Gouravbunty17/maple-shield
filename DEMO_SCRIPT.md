# Maple Shield — Demo Recording Script
**Target length: 2–3 minutes**
**Record with:** OBS Studio (free) or Windows Game Bar (Win+G)

---

## Before You Start (5 min setup)

- [ ] Clean your desktop — close everything except what's needed
- [ ] Open a terminal in `C:\Users\15879\borealis\`
- [ ] Run `run_demo.bat` — wait for both dashboards to open
- [ ] Have a video file ready (drone footage) OR use the simulator (no camera needed)
- [ ] Set screen resolution to 1920×1080
- [ ] Start screen recording

---

## Demo Flow (say this or narrate over it)

### PART 1 — Introduction (0:00–0:20)
> Show the maplesilicon.co/maple-shield website briefly

**Say:**
> "This is Maple Shield — an edge AI platform for real-time drone detection
> and airspace awareness, built by Maple Silicon Inc. in Caledon, Ontario.
> Let me show you the system running live."

---

### PART 2 — Live Detection (0:20–1:10)
> Switch to terminal. Run ONE of the following:

**Option A — with your own video:**
```
python maple_shield_mvp.py --source your_video.mp4
```

**Option B — synthetic scenario (no camera needed, recommended for recording):**
```
python maple_shield_sim.py --scenario incursion
```
> For swarm attack demo:
```
python maple_shield_sim.py --scenario swarm --loops 2
```

**Point out on screen:**
- Bounding boxes appearing around detected objects
- Track IDs staying consistent across frames (e.g. "ID 1", "ID 2")
- Threat level label escalating: **CLEAR → LOW → MEDIUM → HIGH → CRITICAL**
- Threat bar (top-right) lighting up as threat increases
- FPS counter showing real-time performance
- "CoT: ON" indicator confirming C2 output
- JSONL log file being written

**Say:**
> "The system detects objects in real time using a YOLOv8 model trained
> specifically for drone detection — achieving 92.6% mAP accuracy. Each detection
> gets a persistent track ID and a threat score that escalates automatically.
> Simultaneously, alerts stream over MQTT and Cursor on Target — directly
> compatible with ATAK and military command systems."

---

### PART 3 — C2 Simulator (1:10–1:45)
> Switch to browser — http://localhost:5001

**Point out:**
- Live alert feed updating in real time
- Threat level badges (colour coded: green → yellow → orange → red)
- MQTT stream — alerts firing as detections happen
- Timestamp, track ID, threat level on each alert

**Say:**
> "Alerts are streamed in real time over MQTT — compatible with military
> command and control systems including STANAG 4586. Alongside MQTT,
> Maple Shield broadcasts Cursor on Target XML over UDP — making it
> directly readable by ATAK, WinTAK, and TAK Server.
> This is the feed a base operator would see on their screen."

---

### PART 4 — Replay Dashboard (1:45–2:20)
> Switch to browser — http://localhost:5000

**Point out:**
- Timeline of the detection run
- Click on an incident to replay it
- Track history, threat escalation over time
- Evidence log — every frame recorded

**Say:**
> "Every incident is logged and replayable. Operators can review exactly
> what the system detected, when, and how the threat level changed.
> This is the audit trail — critical for defence and law enforcement use."

---

### PART 5 — Close (2:20–2:40)
> Switch back to the website or show the terminal

**Say:**
> "Maple Shield runs on hardware as compact as the NVIDIA Jetson Orin NX —
> under 20 watts. No cloud required. No RF emissions. Fully edge-deployable.
> C2 output over MQTT, Cursor on Target, and STANAG 4586.
> This is a Canadian-built solution for Canadian airspace security.
> Maple Silicon Inc. — Protecting Airspace With Intelligence."

---

## After Recording

- [ ] Export as MP4, 1080p
- [ ] Upload to YouTube (unlisted) or Vimeo
- [ ] Add the link to your IDEaS / Sentinel Shield application
- [ ] Add a "Watch Demo" button to maplesilicon.co/maple-shield

---

## Scenario Quick Reference

| Scenario | Command | Best for |
|---|---|---|
| Single drone incursion | `python maple_shield_sim.py --scenario incursion` | General demo, shows full threat escalation |
| Swarm attack | `python maple_shield_sim.py --scenario swarm` | Sentinel Shield — multi-target tracking |
| Recon orbit | `python maple_shield_sim.py --scenario recon` | Shows persistent tracking, loiter detection |
| Standoff + approach | `python maple_shield_sim.py --scenario standoff` | Shows threat scoring nuance |
| Loop multiple times | Add `--loops 3` to any command | Longer recording |

---

## Tips for a Clean Recording

| Do | Don't |
|---|---|
| Speak slowly and clearly | Rush through screens |
| Zoom in on key numbers (92.6%, threat levels, CoT ON) | Leave irrelevant windows visible |
| Pause briefly when switching windows | Use a noisy background |
| Show the terminal output (it looks technical and credible) | Edit too heavily — keep it authentic |
| Run the `swarm` scenario for maximum visual impact | Use webcam if you don't have drone footage |
