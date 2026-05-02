# Cairn-Edge

**Defence-grade passive vision C-UAS detection node for NVIDIA Jetson Orin Nano.**

Cairn-Edge is the Jetson-optimized edge deployment profile for the CAIRN detection engine behind Maple Shield.

It is designed as a calm, low-power perimeter node:

- vision-only passive detection by default
- no cloud dependency
- DDIL-ready local operation
- CoT/MQTT/REST northbound hooks
- SQLite + JSONL audit store
- Jetson Orin Nano performance constraints treated as hard design limits

> Scope note: Cairn-Edge is a detection, tracking, classification, audit, and interoperability system. It does not implement weapon control or autonomous engagement.

---

## Hardware target

Primary target:

- NVIDIA Jetson Orin Nano 8GB
- 15 W MAXN, with 10 W / 7 W degraded profiles
- 5–6 GB usable memory after OS and services
- NVDEC available for RTSP decode
- no NVENC, so video evidence export is CPU-expensive unless offloaded

---

## System pipeline

```text
RTSP / USB Camera
      ↓
GStreamer ingest
      ↓
NVDEC decode where available
      ↓
Sky ROI + motion prefilter
      ↓
TensorRT INT8 YOLO detector on selected frames/tiles
      ↓
ByteTrack-style CPU tracker + Kalman prediction
      ↓
On-demand verifier for ambiguous tracks
      ↓
CAIRN risk engine
      ↓
SQLite / JSONL / CoT / MQTT / REST / VMS hooks
```

---

## Repository additions

```text
cairn_edge/
├── README.md
├── configs/
│   ├── edge.node.yaml
│   └── roe.default.yaml
├── deepstream/
│   ├── deepstream_2stream_reference.txt
│   └── deepstream_4stream_reference.txt
├── docs/
│   ├── FIELD_INSTALLATION_GUIDE.md
│   ├── OPERATOR_RUNBOOK.md
│   └── RED_TEAM_TEST_PLAN.md
├── scripts/
│   ├── build_trt_engine.py
│   ├── benchmark_harness.py
│   └── jetson_profile_check.py
└── src/
    ├── edge_config.py
    ├── motion_prefilter.py
    ├── track_kinematics.py
    ├── local_store.py
    └── c2_emitters.py
```

---

## Realistic performance targets

| Target | Contractual goal | Engineering note |
|---|---:|---|
| RTSP streams | 2–4 × 1080p @ 15–30 FPS | Requires motion/tile gating and INT8 TensorRT |
| 4K stream | 1 × 4K @ 10–15 FPS | Use upper-sky tiling and strict ROI skip |
| Photon-to-track latency | <400 ms | Measure at frame timestamp, not just inference latency |
| Simultaneous tracks | ≥32 per node | CPU tracker must stay lightweight |
| Power | <12 W steady in 15 W mode | Degrade under thermal pressure |
| Cold start | <60 seconds | Prebuilt TensorRT engines only in production |

Range goals depend on optics, mounting, weather, contrast, sensor quality, and target aspect. Compute alone does not produce 3–10 km detection range.

---

## Run local smoke checks

```bash
python cairn_edge/scripts/jetson_profile_check.py
python cairn_edge/scripts/benchmark_harness.py --synthetic
python cairn_demo.py --config configs/cairn.default.json
```

---

## Production rule

Do not declare field readiness until all are true:

1. TensorRT INT8 detector engine built and benchmarked on target Orin Nano.
2. Latency measured end-to-end, not only model inference.
3. Thermal throttling tested for at least 2 hours.
4. Pd/Pfa measured on a declared test set with weather labels.
5. CoT output validated in ATAK/WinTAK/TAK Server.
6. Audit logs are replayable and signed model artifacts are verified.
