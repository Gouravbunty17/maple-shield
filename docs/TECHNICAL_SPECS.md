# BOREALIS V1 — Technical Specifications

---

## Module APIs

### Detection Module

**Input:**
```python
frame: np.ndarray  # (H, W, 3) BGR uint8
```

**Output:**
```python
detections: List[Dict]
# Each detection:
{
    "cls": int,           # Class ID (0-79 for COCO80)
    "label": str,         # Class name
    "conf": float,        # Confidence [0,1]
    "box": [int, int, int, int]  # [x1, y1, x2, y2]
}
```

**Performance:**
- Inference: 130-150ms (CPU)
- Throughput: 7-8 FPS
- Model size: 6.2 MB (ONNX)

---

### Tracking Module

**API:**
```python
tracker.step(det_list, frame_id, timestamp)
# Returns: det_list with added fields
```

**Added fields:**
```python
{
    "track_id": int,              # Unique persistent ID
    "track_confirmed": bool,      # True if hits >= 2
    "velocity": {
        "vx": float,              # Pixels/frame horizontal
        "vy": float,              # Pixels/frame vertical
        "speed": float            # Pixels/frame magnitude
    },
    "persistence_count": int      # Consecutive high-risk frames
}
```

**Configuration:**
```python
TRACK_IOU_THRES = 0.30      # Matching threshold
TRACK_MAX_AGE = 30          # Frames before deletion
TRACK_MIN_HITS = 2          # Hits for confirmation
TRACK_SMOOTH_ALPHA = 0.70   # EMA smoothing factor
VELOCITY_HISTORY_LENGTH = 5  # Frames for velocity estimation
```

---

### Risk Assessment Module

**API:**
```python
compute_risk_score(box, conf, frame_w, frame_h, velocity_data, persistence_count)
# Returns: risk_data dict
```

**Output:**
```python
{
    "risk_score": float,          # [0,1]
    "risk_state": str,            # "SAFE" | "CAUTION" | "UNSAFE"
    "in_forward_sector": bool,
    "distance_category": str,     # "FAR" | "MEDIUM" | "CLOSE"
    "confidence": float,
    "velocity_score": float,      # [0,1]
    "heading_score": float,       # [0,1] approach detection
    "persistence_frames": int
}
```

**Scoring weights:**
```python
WEIGHT_SECTOR = 0.25      # Forward sector presence
WEIGHT_DISTANCE = 0.25    # Proximity
WEIGHT_CONFIDENCE = 0.15  # Detection quality
WEIGHT_VELOCITY = 0.20    # Speed
WEIGHT_HEADING = 0.15     # Approach direction
# Persistence: +0.2 max escalation for sustained threats
```

**Thresholds:**
```python
UNSAFE_THRESHOLD = 0.7
CAUTION_THRESHOLD = 0.4
VELOCITY_HIGH = 15.0  # px/frame
VELOCITY_MED = 5.0    # px/frame
PERSISTENCE_FRAMES = 30  # Escalation trigger
```

---

## Logging Schema

### JSONL Record (per frame)
```json
{
    "ts": 1738972345.123,
    "frame": 452,
    "infer_ms": 142.3,
    "fps_est": 7.8,
    "max_risk_score": 0.85,
    "max_risk_state": "UNSAFE",
    "detections": [
        {
            "cls": 0,
            "label": "person",
            "conf": 0.89,
            "box": [120, 200, 450, 680],
            "track_id": 7,
            "track_confirmed": true,
            "velocity": {"vx": 3.2, "vy": -1.5, "speed": 3.5},
            "persistence_count": 14,
            "risk": {
                "risk_score": 0.85,
                "risk_state": "UNSAFE",
                "in_forward_sector": true,
                "distance_category": "CLOSE",
                "confidence": 0.89,
                "velocity_score": 0.0,
                "heading_score": 0.8,
                "persistence_frames": 14
            }
        }
    ]
}
```

### Lifecycle Event
```json
{
    "track_id": 7,
    "class": 0,
    "label": "person",
    "birth_frame": 100,
    "birth_timestamp": 1738972332.5,
    "death_frame": 452,
    "death_timestamp": 1738972345.1,
    "lifetime_frames": 352,
    "lifetime_seconds": 12.6,
    "avg_confidence": 0.87,
    "final_velocity_px_per_frame": {"vx": 3.2, "vy": -1.5, "speed": 3.5},
    "max_persistence": 18
}
```

---

## Configuration Parameters

### Detection
```python
MODEL_PATH = "models/yolov8n.onnx"
IMGSZ = 640
CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_DET = 50
```

### Video Recording
```python
SAVE_RAW_VIDEO = True
SAVE_OVERLAY_VIDEO = True
VIDEO_FPS = 30.0
LOG_EVERY_N_FRAMES = 1
```

### Tracking
```python
TRACK_IOU_THRES = 0.30
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 2
TRACK_SMOOTH_ALPHA = 0.70
VELOCITY_HISTORY_LENGTH = 5
```

### Risk Assessment
```python
FORWARD_SECTOR_WIDTH = 0.6
CLOSE_THRESHOLD = 0.15
MEDIUM_THRESHOLD = 0.05
UNSAFE_THRESHOLD = 0.7
CAUTION_THRESHOLD = 0.4
VELOCITY_HIGH = 15.0
VELOCITY_MED = 5.0
PERSISTENCE_FRAMES = 30
```

---

## Performance Metrics

### CPU Baseline (Windows 10, Intel i7)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Frame capture | 2-5ms | CV2 VideoCapture |
| Preprocessing | 3-5ms | Resize + normalize |
| ONNX inference | 130-150ms | YOLOv8n CPU provider |
| Post-processing (NMS) | 2-3ms | NumPy implementation |
| Tracking update | 1-2ms | IoU matching |
| Risk computation | <1ms | Vectorized NumPy |
| Logging (JSONL) | <1ms | Buffered writes |
| Video encoding | 5-10ms | mp4v codec |
| **Total pipeline** | **140-175ms** | **~7-8 FPS** |

### Expected Jetson Performance

| Hardware | Provider | Expected FPS | Expected Latency |
|----------|----------|--------------|------------------|
| CPU baseline | ONNX CPU | 10-15 | ~100ms |
| GPU ONNX | CUDA | 20-25 | ~50ms |
| TensorRT FP16 | TensorRT | 30-35 | ~35ms |

---

## Disk Usage

### Per 1000 frames (~2 minutes @ 8 FPS)

| File | Size | Description |
|------|------|-------------|
| raw.mp4 | ~50 MB | Uncompressed sensor data |
| overlay.mp4 | ~50 MB | Annotated frames |
| detections.jsonl | ~500 KB | Frame-by-frame logs |
| lifecycle.json | ~10 KB | Track statistics |
| meta.json | ~1 KB | Run configuration |
| **Total** | **~100 MB** | Per run |

---

## Dependencies
```
numpy>=1.21.0
opencv-python>=4.5.0
onnxruntime>=1.12.0
pillow>=9.0.0
scikit-image>=0.19.0
```

**Jetson-specific (GPU):**
```
onnxruntime-gpu  # JetPack-specific wheel
```

**Optional (training):**
```
ultralytics>=8.0.0  # YOLOv8 training
```

---

## Error Handling

### Camera Failure
```python
if not cap.isOpened():
    raise RuntimeError("Camera not detected")
```

### Model Loading
```python
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
```

### Disk Space
- No automatic cleanup (operator responsibility)
- Recommend 10GB free for typical mission
- Monitor with `df -h` (Linux) or Task Manager (Windows)

---

## Security Considerations

**Data at rest:**
- JSONL logs contain detection metadata (no PII)
- Videos may contain identifiable persons
- No encryption implemented (operator responsibility)

**Data in transit:**
- Not applicable (offline system)

**Model security:**
- ONNX models are transparent (not obfuscated)
- Model integrity verification not implemented
- Recommend SHA256 checksum validation

---

## Compliance

**Sovereignty:**
- ✅ No cloud dependencies
- ✅ No telemetry
- ✅ No foreign API calls
- ✅ Complete on-device processing

**Audit trail:**
- ✅ Frame-by-frame logging
- ✅ Track lifecycle events
- ✅ Configuration capture
- ✅ Timestamp accuracy (system clock)
- ⏳ Frame hash verification (optional, not enforced)

---

**Last Updated:** 2026-02-08  
**Version:** 1.0 (Phase-1 CPU baseline)
