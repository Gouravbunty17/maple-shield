# Maple Shield — Arctic Training Data Strategy

---

## Problem Statement

**Current limitation:** YOLOv8n trained on COCO80 dataset
- Clear weather, urban/indoor scenes, good lighting
- Performance degrades catastrophically in Arctic conditions:
  - Fog (visibility < 100m)
  - Snow (active precipitation)
  - Whiteout (contrast < 20%)
  - Low-light (polar night, twilight)
  - Sea spray icing

**Required classes for Arctic operations:**
- Ships (commercial, military, icebreakers)
- Icebergs (various sizes, lighting conditions)
- Humans (cold-weather gear, low contrast)
- Vehicles (snowmobiles, tracked vehicles, aircraft)
- Infrastructure (buildings, towers, markers)

**Target:** 5000+ labeled Arctic images across all weather conditions

---

## Three-Track Acquisition Strategy

### Track 1: Partnership with Canadian Institutions

**Target organizations:**

**DRDC (Defence Research and Development Canada)**
- Arctic R&D division
- Declassified maritime patrol imagery
- Contact: Arctic research programs

**Canadian Coast Guard**
- Arctic patrol vessels (imagery archive)
- Ice monitoring datasets
- Search and rescue training footage

**Parks Canada**
- Arctic national parks monitoring
- Wildlife/infrastructure imagery
- Research station cameras

**Environment Canada**
- Arctic weather station cameras
- Ice condition monitoring
- Climate research datasets

**Approach strategy:**
1. Draft 1-page capability brief (sovereignty angle)
2. Propose data-sharing agreement (not purchase)
3. Offer to share trained model (non-exclusive)
4. Emphasize edge deployment (no cloud dependencies)

**Timeline:** 1-3 months (bureaucracy)  
**Expected yield:** 500-2000 images (if successful)

---

### Track 2: Synthetic Data Generation

**Status:** Proof-of-concept validated (augmentation pipeline working)

**Phase 1: Augmentation (COMPLETE)**
- ✅ Fog overlay (density 0.2-0.8)
- ✅ Snow particles (intensity 0.1-0.7)
- ✅ Cold color grading (6000-8000K)
- ✅ Contrast reduction (whiteout effect)
- ✅ Low-light simulation (polar night)
- ✅ 4× multiplier validated (7 → 28 images)

**Phase 2: 3D Rendering Pipeline (PLANNED)**

**Tools:**
- Blender (free, Python scriptable)
- NVIDIA Omniverse (physics-accurate, domain randomization)
- Unity Perception (synthetic dataset generation)

**Assets needed:**
- 3D ship models (public domain sources)
- Iceberg models (procedural generation)
- Arctic environment HDRI (lighting)
- Weather particle systems (fog, snow)

**Randomization parameters:**
```python
Weather:
  - Fog density: 0.0 → 1.0
  - Snow intensity: 0.0 → 1.0
  - Visibility: 10m → 10km
  - Wind speed: 0 → 50 knots

Lighting:
  - Time of day: polar night → midnight sun
  - Cloud cover: 0% → 100%
  - Sun angle: 0° → 30° (Arctic latitude)

Camera:
  - Distance: 10m → 5km
  - Angle: -30° → +30° pitch
  - Height: 1m → 100m above water

Objects:
  - Ship types: cargo, tanker, icebreaker, patrol
  - Iceberg sizes: growler → small island
  - Positions: random within scene
```

**Expected output:**
- 10,000 synthetic images
- Auto-labeled bounding boxes (ground truth)
- 2-3 weeks compute time (single GPU)

**Timeline:** 4-6 weeks (setup + generation)  
**Expected yield:** 10,000+ labeled images

---

### Track 3: Public Dataset Mining + Augmentation

**Existing datasets:**

**1. SeaShips (7000+ ship images)**
- Source: https://github.com/kskin/data
- License: Creative Commons
- Has: Multiple ship types, various weather
- Missing: Arctic-specific conditions
- **Action:** Download + apply Arctic augmentation → 28,000 images

**2. MaritimeNet**
- Source: https://github.com/maritime-net
- Has: Commercial vessels, some adverse weather
- Missing: Extreme Arctic conditions
- **Action:** Augment subset → 5,000 images

**3. NOAA/NASA Polar Archives**
- Source: NOAA National Ice Center, NASA Worldview
- Has: Satellite iceberg imagery
- Missing: Labeled bounding boxes
- **Action:** Manual labeling required (500-1000 images)

**4. Unsplash/Pexels (Public domain)**
- Source: Free stock photo sites
- Search terms: "cargo ship", "arctic", "iceberg", "tanker"
- **Action:** Download 100-200 curated images, augment → 400-800 images

**Timeline:** 1-2 weeks  
**Expected yield:** 33,000+ augmented images

---

## Labeling Strategy

**Tools:**

**CVAT (Computer Vision Annotation Tool)**
- Open source, supports YOLO format
- Assisted labeling (pre-trained model suggests boxes)
- Cloud version: https://app.cvat.ai
- Local Docker: `docker run -p 8080:8080 cvat/server`

**Workflow:**
1. Upload augmented images to CVAT
2. Use YOLOv8n (current model) to auto-generate boxes
3. Human corrects/validates (much faster than scratch)
4. Export in YOLO format
5. Version control with Git LFS

**Labeling rate:**
- From scratch: 10-20 images/hour
- Assisted (pre-trained suggestions): 50-100 images/hour

**Labor estimate:**
- 1000 images assisted: ~10-20 hours
- 5000 images assisted: ~50-100 hours

---

## Training Pipeline

**Phase 1: Fine-tune on augmented data**
```python
from ultralytics import YOLO

# Start from YOLOv8n pretrained weights
model = YOLO('yolov8n.pt')

# Fine-tune on Arctic augmented data
results = model.train(
    data='arctic_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cpu',  # or '0' for GPU
    patience=20,
    save=True
)

# Export to ONNX for Maple Shield
model.export(format='onnx', simplify=True)
```

**Expected performance:**
- mAP@0.5: 0.6-0.7 (vs 0.3-0.4 with COCO80)
- Fog detection: 2-3× improvement
- Snow detection: 2-3× improvement

**Phase 2: Fine-tune on real Arctic data (when available)**

---

## Validation Strategy

**Test set composition:**
- 20% held-out augmented images
- Real Arctic images (if acquired from partners)
- Stratified by weather condition:
  - Clear: 20%
  - Light fog: 20%
  - Moderate fog: 20%
  - Heavy fog/snow: 20%
  - Polar night: 20%

**Metrics:**
- mAP@0.5 (mean average precision)
- Precision/recall per class
- Performance vs weather severity
- False positive rate

**Acceptance criteria:**
- mAP@0.5 > 0.6 overall
- Precision > 0.7 for "ship" class
- No catastrophic failures (missed ships in moderate fog)

---

## Timeline & Milestones

**Week 1-2: Public dataset acquisition**
- ✅ Download SeaShips dataset
- ✅ Apply Arctic augmentation
- ✅ Organize in YOLO structure

**Week 3-4: Labeling**
- Set up CVAT
- Label 1000 images (assisted)
- Quality check

**Week 5-6: Training v1**
- Train on augmented data
- Validate performance
- Export ONNX model

**Week 7-8: Partnership outreach**
- Draft DRDC proposal
- Contact Coast Guard
- Submit requests

**Week 9-12: Synthetic pipeline (if needed)**
- Set up Blender/Omniverse
- Generate 10,000 synthetic images
- Retrain model

**Month 3-4: Real data integration (if acquired)**
- Receive partner data
- Label real Arctic images
- Fine-tune model v2

---

## Budget Estimate

**Labor:**
- Labeling (100 hours @ contractor rate): $5,000-$10,000
- Or DIY: 100 hours spread over 4 weeks

**Compute:**
- Training on CPU: Free (slow)
- Training on cloud GPU (AWS p3.2xlarge): $100-$200
- Synthetic generation (local GPU): Free

**Data acquisition:**
- Public datasets: Free
- Partnership data: Free (if successful)
- Paid datasets (fallback): $1,000-$5,000

**Total:** $0-$15,000 depending on approach

---

## Risk Mitigation

**Risk:** Partnerships fail to provide data  
**Mitigation:** Synthetic + augmentation pipeline already proven

**Risk:** Augmented data doesn't transfer to real Arctic  
**Mitigation:** Validation on held-out real images (if acquired)

**Risk:** Labeling takes too long  
**Mitigation:** Assisted labeling, phased approach (1000 → 5000)

**Risk:** Model performance insufficient  
**Mitigation:** Iterative improvement, ensemble methods

---

## Success Metrics

**Phase 1 (Augmentation only):**
- 33,000+ augmented Arctic images
- mAP@0.5 > 0.5 on augmented test set
- Qualitative improvement visible

**Phase 2 (+ Real data):**
- 500+ real Arctic images labeled
- mAP@0.5 > 0.6 overall
- Validated on real Arctic imagery

**Phase 3 (+ Synthetic):**
- 10,000+ synthetic images
- mAP@0.5 > 0.7 overall
- Ready for field deployment

---

**Last Updated:** 2026-02-08  
**Status:** Track 3 (augmentation) validated, Tracks 1-2 planned


