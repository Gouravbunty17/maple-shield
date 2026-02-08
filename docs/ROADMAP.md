# BOREALIS — Development Roadmap

---

## Phase 1: CPU Validation ✅ COMPLETE

**Objective:** Prove pipeline architecture on development hardware

**Deliverables:**
- ✅ YOLOv8n ONNX inference (CPU)
- ✅ IoU-based tracking with velocity estimation
- ✅ Motion-weighted risk assessment
- ✅ Defense-grade JSONL logging
- ✅ Raw/overlay video separation
- ✅ JSON-first replay tool
- ✅ Arctic augmentation pipeline (proof-of-concept)

**Performance achieved:**
- 7-8 FPS sustained
- 130-150ms inference latency
- Stable tracking across 800+ frames
- Complete audit trail

**Status:** ✅ Validated on Windows 10, Intel i7

---

## Phase 2: Jetson Migration ⏳ PENDING HARDWARE

**Objective:** Prove 2-3× performance on Arctic-ready edge platform

**Hardware:** Jetson Orin Nano Developer Kit

**Milestones:**

**2.1: Environment Setup (1 week)**
- [ ] JetPack installation/verification
- [ ] Python environment setup
- [ ] Dependency installation (ONNX Runtime GPU)
- [ ] Camera validation
- [ ] Code transfer from Windows

**2.2: CPU Baseline (1 week)**
- [ ] Run pipeline on Jetson CPU
- [ ] Benchmark vs Windows CPU
- [ ] Thermal profiling
- [ ] Expected: 10-15 FPS

**2.3: GPU Acceleration (1 week)**
- [ ] Enable CUDA provider
- [ ] Benchmark GPU vs CPU
- [ ] Power consumption measurement
- [ ] Expected: 20-25 FPS (2× improvement)

**2.4: TensorRT Optimization (2 weeks)**
- [ ] Convert ONNX → TensorRT engine
- [ ] FP16 precision validation
- [ ] Benchmark TensorRT vs GPU ONNX
- [ ] Thermal throttle testing
- [ ] Expected: 30-35 FPS (3× improvement)

**Success criteria:**
- ✅ Pipeline ports without code changes
- ✅ TensorRT achieves ≥30 FPS
- ✅ Thermal stability under continuous operation
- ✅ Logging/replay infrastructure works identically

**Timeline:** 4-6 weeks (when hardware available)

---

## Phase 3: Arctic Training Data ⏳ IN PROGRESS

**Objective:** Train Arctic-specific detection model

**Current status:**
- ✅ Augmentation pipeline validated (7 → 28 images)
- ⏳ Public dataset mining (SeaShips target)
- ⏳ Partnership outreach (DRDC, Coast Guard)
- ⏳ Synthetic pipeline design (Blender/Omniverse)

**Milestones:**

**3.1: Augmented Dataset (4 weeks)**
- [ ] Download SeaShips (7000 images)
- [ ] Apply Arctic augmentation (→ 28,000 images)
- [ ] Label 1000 images (CVAT assisted)
- [ ] Train YOLOv8 Arctic v1

**3.2: Partnership Data (2-4 months)**
- [ ] Draft DRDC proposal
- [ ] Contact Coast Guard, Parks Canada
- [ ] Negotiate data-sharing agreements
- [ ] Receive declassified imagery
- [ ] Label real Arctic images
- [ ] Train YOLOv8 Arctic v2

**3.3: Synthetic Pipeline (6 weeks)**
- [ ] Set up Blender rendering
- [ ] Acquire 3D ship/iceberg models
- [ ] Build weather randomization
- [ ] Generate 10,000 synthetic images
- [ ] Train YOLOv8 Arctic v3

**Success criteria:**
- ✅ mAP@0.5 > 0.6 on Arctic test set
- ✅ 2-3× improvement vs COCO80 in fog
- ✅ Validated on real Arctic imagery (if available)

**Timeline:** 3-6 months (parallel tracks)

---

## Phase 4: Inference Scheduler (HIGH-RISK IP) ⏳ PLANNED

**Objective:** Power-aware, thermal-aware frame processing

**Motivation:**
- Arctic deployment: battery-limited, thermal-constrained
- Not all frames equally important (empty sea vs active threat)
- Need graceful degradation under throttle

**Design:**

**4.1: Threat-based priority queue**
```python
Priority levels:
  - P0: Active UNSAFE threats (process every frame)
  - P1: CAUTION objects (process every 2 frames)
  - P2: SAFE / empty (process every 5 frames)
  - P3: Idle (process every 10 frames)
```

**4.2: Thermal throttle handling**
```python
if temp > 75°C:
    frame_skip = 2  # Process every other frame
if temp > 80°C:
    frame_skip = 5  # Process every 5th frame
if temp > 85°C:
    pause_inference()  # Cooldown mode
```

**4.3: Battery-aware scheduling**
```python
if battery < 20%:
    reduce_fps(target=15)  # Extend runtime
if battery < 10%:
    reduce_fps(target=5)   # Emergency mode
```

**Milestones:**

**4.1: Thermal profiling (2 weeks)**
- [ ] Measure Jetson temps under load
- [ ] Identify throttle points
- [ ] Characterize cooldown curves

**4.2: Priority queue implementation (2 weeks)**
- [ ] Risk-based frame selection
- [ ] Adaptive frame skip logic
- [ ] Validation against full-rate

**4.3: Battery integration (1 week)**
- [ ] Read battery state (if available)
- [ ] Implement power-saving modes
- [ ] Test runtime extension

**Success criteria:**
- ✅ 2× battery life extension in low-threat scenarios
- ✅ No missed threats during thermal throttle
- ✅ Graceful degradation under constraints

**Timeline:** 6-8 weeks

---

## Phase 5: Field Validation ⏳ PLANNED

**Objective:** Prove system in representative Arctic conditions

**Test scenarios:**

**5.1: Weather conditions**
- [ ] Clear visibility baseline
- [ ] Light fog (visibility 500m-1km)
- [ ] Moderate fog (visibility 100-500m)
- [ ] Heavy fog (visibility <100m)
- [ ] Active snow
- [ ] Polar twilight (low-light)

**5.2: Target detection**
- [ ] Commercial vessel at 1km, 2km, 5km
- [ ] Small boat at 500m, 1km
- [ ] Human on ice at 100m, 200m
- [ ] Snowmobile at 500m, 1km

**5.3: Environmental stress**
- [ ] Temperature: -20°C to +10°C
- [ ] Wind: 0-40 knots
- [ ] Vibration (vessel motion)
- [ ] Sea spray (lens contamination)

**Validation metrics:**
- Detection rate vs range
- False positive rate
- Thermal stability (continuous 8-hour operation)
- Battery life
- Replay integrity (audit trail verification)

**Timeline:** 2-4 weeks (requires Arctic access)

---

## Phase 6: Sovereignty Evaluation ⏳ PLANNED

**Objective:** Assess hardware alternatives for supply chain resilience

**Current:** Jetson Orin Nano (NVIDIA, US-based)

**Alternatives:**

**RK3588 (Rockchip, China-based)**
- ARM Cortex-A76 + Mali GPU
- NPU for inference acceleration
- Lower cost, wider availability
- Requires custom RKNN model conversion

**x86 + Intel iGPU (US/domestic)**
- Intel NUC or similar
- OpenVINO toolkit for optimization
- Higher power consumption
- Better software compatibility

**Evaluation criteria:**

| Factor | Weight | Jetson | RK3588 | x86+iGPU |
|--------|--------|--------|--------|----------|
| Performance (FPS) | 30% | TBD | TBD | TBD |
| Power efficiency | 20% | TBD | TBD | TBD |
| Supply chain risk | 25% | Medium | High | Low |
| Software maturity | 15% | High | Medium | High |
| Cost | 10% | $$$ | $ | $$ |

**Milestones:**

**6.1: RK3588 evaluation (4 weeks)**
- [ ] Acquire RK3588 dev board
- [ ] Port BOREALIS to RKNN
- [ ] Benchmark performance
- [ ] Assess thermal characteristics

**6.2: x86 evaluation (2 weeks)**
- [ ] Acquire Intel NUC or similar
- [ ] Install OpenVINO
- [ ] Convert model to IR format
- [ ] Benchmark performance

**6.3: Decision matrix (1 week)**
- [ ] Compare all platforms
- [ ] Risk assessment (sovereignty, supply chain)
- [ ] Recommendation report

**Timeline:** 6-8 weeks

---

## Long-term Vision (12-24 months)

**Multi-sensor fusion:**
- Radar integration (all-weather detection)
- Thermal camera (polar night capability)
- Lidar (range/depth accuracy)

**Advanced models:**
- Semantic segmentation (ice vs water vs land)
- Object tracking (Kalman filter, BYTETrack)
- Trajectory prediction (collision avoidance)

**Network capabilities:**
- Mesh networking (multi-platform coordination)
- Encrypted data links (secure comms)
- Central command integration (optional cloud sync)

**Autonomy:**
- Waypoint navigation (GPS + vision)
- Obstacle avoidance (reactive control)
- Return-to-base (failsafe mode)

---

## Dependencies & Blockers

**Critical path:**
1. **Jetson hardware** → blocks Phase 2
2. **Arctic training data** → blocks field validation accuracy
3. **Arctic field access** → blocks Phase 5

**Risk mitigation:**
- Synthetic data reduces dependency on real Arctic imagery
- CPU validation proves architecture before hardware arrives
- Augmentation pipeline provides interim model improvements

---

## Resource Requirements

**Hardware:**
- Jetson Orin Nano: $500
- RK3588 dev board: $200 (evaluation)
- Arctic camera housing: $1,000-$5,000
- Power/cooling: $500

**Software/Data:**
- Arctic training data: $0-$15,000 (see data strategy)
- Cloud compute (optional): $100-$500

**Labor:**
- Engineering: 6-12 months (1 FTE)
- Field testing: 2-4 weeks
- Partnership coordination: Ongoing

---

**Last Updated:** 2026-02-08  
**Current Phase:** Phase 1 complete, Phase 2-3 in progress
