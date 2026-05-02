# Maple Shield — powered by CAIRN

**Canadian Sovereign Edge AI Intelligence System.** Lawful, **passive**
vision-first early-warning for drone activity in monitored airspace. CAIRN
is the internal detection / risk engine; Maple Shield is the product.

The platform is observational. It does not intercept, jam, neutralize,
target, or otherwise interfere with drones. See `platform/docs/COMPLIANCE.md`
and `platform/docs/SCOPE.md` for hard product boundaries.

## Quick Links

- **Doc ↔ repo alignment (every claim mapped to evidence):** [docs/DOC_REPO_ALIGNMENT.md](docs/DOC_REPO_ALIGNMENT.md)
- **Demo validation (what the demo proves and does not prove):** [docs/DEMO_VALIDATION.md](docs/DEMO_VALIDATION.md)
- **NATO IC26 repo-aligned response:** [docs/NATO_IC26_REPO_ALIGNED_RESPONSE.md](docs/NATO_IC26_REPO_ALIGNED_RESPONSE.md)
- **NATO IC26 claim alignment matrix:** [docs/NATO_IC26_CLAIM_ALIGNMENT_MATRIX.md](docs/NATO_IC26_CLAIM_ALIGNMENT_MATRIX.md)
- **CAIRN Engine v2:** [docs/CAIRN_ENGINE_V2.md](docs/CAIRN_ENGINE_V2.md)
- **Documentation index:** [docs/README.md](docs/README.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Architecture:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Verified today (proven or demo-supported)

These claims are backed by running code and tests in this repo. See
[docs/DOC_REPO_ALIGNMENT.md](docs/DOC_REPO_ALIGNMENT.md) for per-claim
evidence pointers.

- **Passive vision-first** boundary, enforced in code by
  `platform/tests/test_compliance.py` (forbidden-token grep + zero
  DELETE endpoints + drone-only class set).
- **CAIRN risk engine** with stable detection / risk / frame schemas,
  explainable risk factors, reasons, and recommended operator actions.
- **JSON / audit-ready records** (`CairnFrameRecord.to_json()` schema
  `cairn.frame.v1`) with hash-chained operator audit log in the
  platform `command-api`.
- **Operator console + replay** (`platform/operator-ui/`) with Live,
  Alerts, Replay, and Audit views; React + Vite + TypeScript build is
  clean (176.53 KB JS / 57.15 KB gzipped); npm audit reports 0
  vulnerabilities.
- **Repeatable benchmark** (`scripts/benchmark.py`) measuring CAIRN
  scoring-stage FPS and p50 / p95 / p99 latency on host CPU, emitting
  `frames.jsonl`, `summary.json`, `manifest.json` per run.
- **One-command demo** (`run_maple_shield_demo.py`) with PASS / FAIL
  gating against the benchmark output.
- **Cross-cutting tests**: repo and platform suites cover the benchmark,
  demo runner, CAIRN session helper, browser QA docs, and compliance
  boundaries.

## In progress

- Real `CairnSourceDetector` adapter is shipped in
  `platform/edge-agent/edge_agent/cairn_adapter.py`. Wiring a real YOLO
  ONNX detection provider into the adapter (so live frames flow through
  CAIRN end-to-end) is the next product upgrade. The current demo
  boundary is documented in `docs/DEMO_VALIDATION.md`.

## Roadmap (intentionally not yet claimed)

Each item below has a scaffold or hook in the repo but is **not** backed
by a measurement or live test. Marketing material must label these as
roadmap.

- **STANAG 4609 conformance.** Scaffold in
  `cairn_edge/src/advanced/stanag4609_export.py`.
- **Thermal / IR fusion.** Scaffold in
  `cairn_edge/src/advanced/thermal_fusion.py` and `configs/thermal.yaml`.
- **Multi-node mesh.** Scaffold in
  `cairn_edge/src/advanced/mesh_sync.py`.
- **Arctic-condition field validation.** `arctic_augment.py` exists for
  synthetic augmentation. No real-world Arctic data in repo.
- **<2 % false-positive rate.** No field-test data in repo.
- **<50 ms full detection-to-alert latency on Jetson Orin Nano.** Not
  measured. The benchmark currently measures the **CAIRN scoring stage
  on host CPU** and labels it that way in `summary.json`.
- **Jetson Orin Nano power-budget validation.** No measured numbers.

## Quick start

```powershell
# repeatable demo, no camera or model required
python run_maple_shield_demo.py
# inspect: runs/demo-<timestamp>/summary.json

# benchmark with custom shape
python scripts/benchmark.py --frames 600 --tracks 3

# full test suite
python -m pytest -q tests
cd platform
python -m pytest -q
```

## Smoke test (one frame)

```bash
python cairn_demo.py
```

With explicit config:

```bash
python cairn_demo.py --config configs/cairn.default.json
```

See [docs/CAIRN_ENGINE_V2.md](docs/CAIRN_ENGINE_V2.md) for the next
integration steps and [docs/DOC_REPO_ALIGNMENT.md](docs/DOC_REPO_ALIGNMENT.md)
for the live alignment table.
