# BOREALIS V1 — Deployment Guide

---

## Windows Deployment (Development/Testing)

### Prerequisites

- Windows 10/11
- Python 3.10
- Webcam or USB camera
- 8GB RAM minimum
- 10GB free disk space

### Installation Steps

**1. Install Python dependencies:**
```powershell
pip install numpy opencv-python onnxruntime pillow scikit-image
```

**2. Download BOREALIS:**
```powershell
git clone <repository-url>
cd borealis
```

Or manually download and extract to `C:\Users\<username>\borealis\`

**3. Download YOLOv8n model:**
```powershell
mkdir models -Force
cd models
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx" -OutFile "yolov8n.onnx"
cd ..
```

**4. Verify installation:**
```powershell
python borealis_v1_motion_risk.py
```

Press 'q' to stop after a few seconds. Check `runs\` directory for output.

---

## Jetson Orin Nano Deployment (Production)

### Prerequisites

- Jetson Orin Nano Developer Kit
- JetPack 5.1.2+ or 6.0+
- USB camera or CSI camera
- 64GB microSD card minimum
- Active cooling (fan recommended)

### Installation Steps

**1. SSH into Jetson:**
```bash
ssh <username>@<jetson-ip>
```

**2. System update:**
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0
```

**3. Create project directory:**
```bash
mkdir -p ~/borealis
cd ~/borealis
```

**4. Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**5. Install dependencies:**
```bash
pip install --upgrade pip setuptools wheel
pip install numpy opencv-python pillow scikit-image
```

**6. Install ONNX Runtime (GPU):**

Check JetPack version:
```bash
cat /etc/nv_tegra_release
```

For JetPack 5.x:
```bash
pip install onnxruntime-gpu --extra-index-url https://pypi.nvidia.com
```

For JetPack 6.x:
```bash
pip install onnxruntime-gpu
```

**7. Transfer code from Windows:**

From Windows PowerShell:
```powershell
scp borealis_v1_motion_risk.py <username>@<jetson-ip>:~/borealis/
scp borealis_tracker_v3.py <username>@<jetson-ip>:~/borealis/
scp borealis_risk_v2.py <username>@<jetson-ip>:~/borealis/
scp borealis_v1_replay_json.py <username>@<jetson-ip>:~/borealis/
scp models\yolov8n.onnx <username>@<jetson-ip>:~/borealis/models/
```

**8. Test camera:**
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"
```

**9. Run pipeline (CPU baseline):**
```bash
python3 borealis_v1_motion_risk.py
```

**10. Enable GPU acceleration:**

Edit `borealis_v1_motion_risk.py`, change line:
```python
# FROM:
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# TO:
sess = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

Run again:
```bash
python3 borealis_v1_motion_risk.py
```

Expected performance: ~20-25 FPS (2-3× faster than CPU)

---

## TensorRT Optimization (Maximum Performance)

**1. Install TensorRT tools:**
```bash
pip install polygraphy --extra-index-url https://pypi.nvidia.com
```

**2. Convert ONNX → TensorRT:**
```bash
cd ~/borealis/models
trtexec --onnx=yolov8n.onnx \
        --saveEngine=yolov8n_fp16.engine \
        --fp16 \
        --workspace=4096 \
        --verbose
```

This takes 2-5 minutes. Expected output: `yolov8n_fp16.engine` file.

**3. TensorRT inference script:**

(Requires `tensorrt` Python bindings - advanced setup, contact for implementation)

Expected performance: ~30-35 FPS

---

## Configuration

### Adjust risk sensitivity:

Edit `borealis_risk_v2.py`:
```python
# More sensitive (earlier warnings):
UNSAFE_THRESHOLD = 0.6  # default: 0.7
CAUTION_THRESHOLD = 0.3  # default: 0.4

# Less sensitive (fewer alerts):
UNSAFE_THRESHOLD = 0.8
CAUTION_THRESHOLD = 0.5
```

### Adjust tracking parameters:

Edit `borealis_tracker_v3.py`:
```python
# Longer track persistence:
TRACK_MAX_AGE = 60  # default: 30

# Faster confirmation:
TRACK_MIN_HITS = 1  # default: 2 (not recommended)
```

### Change detection confidence:

Edit `borealis_v1_motion_risk.py`:
```python
# More detections (more false positives):
CONF_THRES = 0.25  # default: 0.35

# Fewer detections (fewer false positives):
CONF_THRES = 0.50
```

---

## Troubleshooting

### Camera not detected

**Windows:**
```powershell
# Try different camera indices
python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened()); cap.release()"
```

**Jetson:**
```bash
# For CSI camera, use gstreamer pipeline instead of index 0
# Edit borealis_v1_motion_risk.py, replace:
cap = cv2.VideoCapture(0)
# With:
cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)
```

### Low FPS on Jetson

**Check if GPU is being used:**
```bash
# During inference, run in separate terminal:
sudo tegrastats
```

Look for "GPU" percentage > 0. If GPU is idle, CUDA provider not working.

**Check ONNX Runtime providers:**
```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Thermal throttling

**Monitor temperature:**
```bash
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'
```

If temps > 80°C, add active cooling (fan) or reduce load.

**Reduce frame rate to lower thermal load:**

Edit `borealis_v1_motion_risk.py`:
```python
# Add frame skip logic
if frame_id % 2 == 0:  # Process every other frame
    continue
```

---

## Mission Workflow

**1. Pre-flight checklist:**
- [ ] Camera connected and tested
- [ ] Sufficient disk space (10GB+)
- [ ] Cooling active (Jetson)
- [ ] Time sync verified

**2. Start recording:**
```bash
python3 borealis_v1_motion_risk.py
```

**3. Monitor live feed:**
- Watch for risk alerts (RED boxes = UNSAFE)
- Check FPS stability
- Monitor disk usage

**4. Stop recording:**
- Press 'q' when mission complete

**5. Post-mission analysis:**
```bash
# Find latest run
ls -lt runs/ | head -n 2

# Replay mission
python3 borealis_v1_replay_json.py runs/<timestamp>

# Extract high-risk events
grep "UNSAFE" runs/<timestamp>/detections.jsonl | head -n 10
```

---

## Data Management

### Disk usage estimation:

| Duration | Frames (@8 FPS) | Disk Usage |
|----------|-----------------|------------|
| 1 minute | 480 | ~50 MB |
| 10 minutes | 4800 | ~500 MB |
| 1 hour | 28800 | ~3 GB |
| 8 hours | 230400 | ~24 GB |

### Cleanup old runs:
```bash
# List runs by size
du -sh runs/* | sort -h

# Delete runs older than 7 days
find runs/ -type d -mtime +7 -exec rm -rf {} +
```

### Archive to external storage:
```bash
# Compress run
tar -czf mission_20260208.tar.gz runs/20260208_*

# Copy to USB drive
cp mission_20260208.tar.gz /media/usb/
```

---

## Support

**Hardware issues:** Check Jetson forums, NVIDIA Developer Zone  
**Software bugs:** Check project repository issues  
**Performance tuning:** Consult TensorRT optimization guides  

---

**Last Updated:** 2026-02-08  
**Version:** 1.0 (Phase-1 CPU baseline)
