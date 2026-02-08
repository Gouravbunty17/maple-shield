# BOREALIS Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-02-08

### Added
- YOLOv8n ONNX inference engine (CPU provider)
- IoU-based tracking with EMA smoothing (alpha=0.7)
- Velocity estimation (5-frame position history)
- Motion-weighted risk assessment:
  - Forward sector geometry (25% weight)
  - Distance proxy via box area (25% weight)
  - Confidence (15% weight)
  - Velocity magnitude (20% weight)
  - Heading analysis (15% weight)
  - Persistence escalation (up to +0.2)
- Defense-grade JSONL logging (frame-by-frame detections)
- Raw/overlay video separation (forensic replay capability)
- Lifecycle tracking (track birth/death with velocity stats)
- JSON-first replay tool with pause/speed controls
- Arctic augmentation pipeline (fog, snow, cold grading, low-light)
- Complete documentation package (5 documents, 50+ pages)
- System architecture and data flow diagrams

### Performance
- CPU baseline: 7-8 FPS sustained on Intel i7
- Inference latency: 130-150ms per frame
- Track stability: Validated across 800+ frames
- Augmentation: 4× multiplier (7 → 28 images)

### Documentation
- ARCHITECTURE.md — System design and roadmap
- TECHNICAL_SPECS.md — APIs, schemas, configuration
- DEPLOYMENT_GUIDE.md — Installation and operations
- ARCTIC_DATA_STRATEGY.md — Training data acquisition plan
- ROADMAP.md — Development phases and milestones
- Visual diagrams (architecture block + data flow sequence)

### Known Issues
- Heading score is heuristic (not true depth-based approach detection)
- Frame hash verification not enforced (optional)
- Thermal throttle handling not implemented (planned for Phase 4)
- No automated testing framework (manual validation only)

### Dependencies
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- onnxruntime >= 1.12.0
- pillow >= 9.0.0
- scikit-image >= 0.19.0

---

## [Unreleased]

### Planned - Phase 2 (Jetson Migration)
- GPU ONNX Runtime (CUDA provider)
- TensorRT FP16 optimization
- Target: 30-35 FPS on Jetson Orin Nano

### Planned - Phase 3 (Arctic Training Data)
- Fine-tuned YOLOv8 on Arctic ship dataset
- Partnership data from DRDC/Coast Guard
- Synthetic data generation (Blender/Omniverse)

### Planned - Phase 4 (Inference Scheduler)
- Power-aware frame processing
- Thermal throttle handling
- Threat-based priority queue

---

## Version History

- **1.0.0** (2026-02-08): Phase-1 CPU validation complete
- **0.x.x**: Development iterations (not documented)

---

## Versioning Scheme

**MAJOR.MINOR.PATCH** (Semantic Versioning)

- **MAJOR**: Incompatible architecture changes (e.g., new log format)
- **MINOR**: New features, backward compatible (e.g., BYTETrack upgrade)
- **PATCH**: Bug fixes, performance improvements (e.g., NMS optimization)

**Examples:**
- `1.0.0 → 1.1.0`: Add stereo depth estimation (new feature)
- `1.1.0 → 1.1.1`: Fix tracker ID swap bug (patch)
- `1.1.1 → 2.0.0`: Change JSONL schema (breaking change)
