# CAIRN Engine v2

**CAIRN** is the internal detection engine behind **Maple Shield**.

> Maple Shield — powered by the CAIRN detection engine.

CAIRN replaces the older ARGUS concept name and turns the project from a single demo script into a modular edge-AI detection engine with explainable risk scoring, audit-ready outputs, and clean integration points for C2 adapters.

---

## Why this upgrade matters

The previous Maple Shield prototype already proved the Phase-1 pipeline:

1. EO camera input
2. YOLO ONNX detection
3. IoU tracking
4. Threat/risk scoring
5. Overlay video
6. JSONL logs
7. MQTT / CoT demo outputs

That is good for a demo. It is not enough for a defensible engine architecture.

CAIRN v2 adds the missing product layer:

- Stable detection/risk schemas
- Configurable thresholds and weights
- Explainable risk factors and reasons
- Frame-level audit records
- Adapters from existing Maple Shield scripts
- A no-camera smoke demo
- A clean package boundary for future detector, tracker, sensor-fusion, health, and C2 modules

---

## New package structure

```text
cairn_engine/
├── __init__.py          # Public CAIRN package exports
├── adapters.py          # Converts existing Maple Shield detections into CAIRN format
├── config.py            # Engine and risk configuration dataclasses
├── engine.py            # Frame-level CAIRN engine wrapper
├── risk_engine.py       # Explainable configurable risk engine
└── schemas.py           # Stable CAIRN detection/risk/frame records

configs/
└── cairn.default.json   # Default operational config

cairn_demo.py            # No-camera smoke test
```

---

## Current CAIRN data flow

```text
Detector / Tracker Output
        ↓
CAIRN Adapter
        ↓
CairnDetection schema
        ↓
CairnRiskEngine
        ↓
CairnRiskResult
        ↓
CairnFrameRecord
        ↓
JSONL audit / overlay / MQTT / CoT / replay / future C2
```

---

## Risk factors

CAIRN v2 scores each tracked object using configurable factors:

| Factor | Meaning |
|---|---|
| object_type | Drone, bird, unknown, ignored class |
| confidence | Model confidence |
| track | Whether the track is confirmed/persistent |
| zone | Whether it is inside the protected zone |
| distance | Bounding-box area as a short-range proxy |
| velocity | Pixel motion rate until calibrated geometry is added |
| persistence | Sustained presence across frames |

Each result includes:

- `risk_score`
- `threat_level`
- `factors`
- `reasons`
- `recommended_operator_action`

This makes CAIRN easier to explain to defence reviewers than a black-box detector.

---

## Run smoke demo

```bash
python cairn_demo.py
```

With config file:

```bash
python cairn_demo.py --config configs/cairn.default.json
```

Expected output: a JSON frame record with two demo tracks, one drone-like and one bird-like.

---

## Next engineering steps

### Step 1 — Wire CAIRN into `maple_shield_mvp.py`

Replace direct threat scoring calls with:

```python
from cairn_engine import CairnEngine
from cairn_engine.adapters import batch_from_mvp

cairn = CairnEngine()
...
cairn_detections = batch_from_mvp(det_list, fw, fh)
record = cairn.process_frame(
    frame_id=frame_id,
    detections=cairn_detections,
    session_id=run_dir.name,
    frame_w=fw,
    frame_h=fh,
    fps=fps,
    infer_ms=infer_ms,
)
```

Then use `record.to_json()` for JSONL logging and pass `record.risks` into overlay / MQTT / CoT adapters.

### Step 2 — Add health monitor

Add runtime health fields:

- camera status
- model provider
- GPU/CPU provider
- FPS rolling average
- frame-drop count
- disk-free estimate
- thermal status when running on Jetson

### Step 3 — Add model registry

Move model metadata into config:

- model path
- labels path
- input size
- provider priority
- checksum
- training dataset version

### Step 4 — Add geometry calibration

Current velocity and range are pixel-based. For serious field credibility, add camera calibration:

- camera horizontal FOV
- vertical FOV
- mounting height
- approximate bearing calculation
- optional range estimate from known target size

### Step 5 — Add multi-node schema

Prepare for mesh coordination:

- node ID
- sensor ID
- camera position
- bearing sector
- handoff-ready track metadata

---

## Naming rule

Use **ARGUS** only as legacy/internal history. Public and repo-facing language should now use:

- **CAIRN detection engine**
- **CAIRN Engine v2**
- **Maple Shield — powered by CAIRN**

Avoid calling the product itself CAIRN. The product is **Maple Shield**. CAIRN is the engine.
