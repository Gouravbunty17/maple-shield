# NATO IC26 Claim Alignment Matrix

Source document reviewed: `C:\Users\15879\Downloads\Maple Shield NATO IC26 Final.pdf`

Repo evidence baseline: PR #13 (`codex/phase3-proof-layer`, commit `228b8d5`).

## Decision Rule

Use these labels consistently:

- **Keep as proven**: running repo code plus tests or reproducible measurement.
- **Keep as demo-supported**: deterministic demo supports the claim, but field
  conditions are not proven.
- **Downgrade to in progress**: scaffold or partial code exists, but the full
  capability is not exercisable end-to-end.
- **Downgrade to roadmap**: no measurement or test exists yet.
- **Remove/reframe**: claim is too absolute, too broad, or outside the passive
  product boundary.

## High-Impact Alignment Changes

| Original PDF claim | Repo-aligned status | Replacement wording |
|---|---|---|
| TRL 4 -> TRL 5 outdoor prototype testing in progress | Downgrade to in progress | Software proof layer is validated in repo; outdoor prototype and live-drone testing remain IC26 validation objectives. |
| Detects, tracks, and classifies Class I, Class II, and micro UAS threats | Demo-supported/in progress | Intended coverage includes micro, Class I, and Class II UAS; current proof validates CAIRN scoring over deterministic drone detections. |
| 10-50x more detection nodes | Roadmap | Lower-cost passive nodes are the intended economic wedge; deployment-density advantage requires hardware and field cost validation. |
| Zero electromagnetic footprint, invisible to ELINT/SIGINT | Reframe | Passive sensing does not require radar/RF detection; networking and deployment choices may still emit. |
| Less than $1K per node | Roadmap | Cost target, not repo-proven. |
| 15 km Arctic perimeter, autonomous mesh, -40 C | Roadmap | Arctic and mesh validation are requested IC26 trial objectives. |
| SparseFlow delivers sub-25 W edge inference | Roadmap for Maple Shield | SparseFlow is a company moat; Maple Shield repo has no measured power result. |
| 1.68x A100 speedup and 31.47 TFLOPS CUDA result | External/company evidence, not Maple Shield repo evidence | Can mention only as Maple Silicon background if separately documented; do not present as Maple Shield field performance. |
| Detection range 300-800 m / 800-1500 m | Roadmap | Range must be measured in controlled live tests. |
| Thermal/IR 500-1200 m | Roadmap | Thermal/IR fusion scaffold exists; no validation evidence. |
| mAP >90 percent / >85 percent degraded | Roadmap | No benchmark evaluation artifacts in repo. |
| False positive rate <2 percent | Roadmap | No labelled field evaluation in repo. |
| 30+ FPS on Jetson Orin Nano | Roadmap | PR #13 measures CAIRN scoring stage on host CPU only. |
| Detection-to-alert latency <50 ms end-to-end | Roadmap | PR #13 measures CAIRN scoring-stage latency only, not full pipeline. |
| Power consumption <25 W per node | Roadmap | No power measurement in repo. |
| Operating temperature -40 C to +50 C | Roadmap | No environmental validation in repo. |
| Multi-object tracking handles 20+ simultaneous targets per node | Roadmap/in progress | Platform supports tracking concepts; no benchmark demonstrates 20+ live targets per node. |
| Mesh coordination across nodes | Roadmap | Scaffold exists, not exercised by demo/test. |
| STANAG-compatible data feeds | Roadmap | Structured JSONL is proven; STANAG conformance is not validated. |
| Validated for -40 C | Remove/downgrade | Replace with "cold-weather validation requested through IC26." |
| Completely immune to electronic attack | Remove/reframe | Replace with "not dependent on RF/GPS sensing; optical/weather/power/networking limits remain." |
| Next Generation Targeting, faster kill-chain closure | Reframe | Replace with "operator awareness and cueing metadata for authorized downstream systems." |
| Effector cueing and engagement prioritization | Reframe | Replace with "operator alert prioritization and structured track metadata." |
| Deployment-ready within 18 months | Keep only as target | "Target, dependent on IC26 validation and measured field performance." |

## Claims Supported by PR #13

| Claim | Evidence |
|---|---|
| Passive observation boundary | `platform/docs/COMPLIANCE.md`, `platform/tests/test_compliance.py`. |
| No DELETE endpoints / append-only posture | `platform/tests/test_compliance.py::test_no_delete_endpoint_anywhere`. |
| CAIRN frame records | `cairn_engine.schemas.CairnFrameRecord.to_json()`, `tests/test_benchmark.py`. |
| CAIRN scoring benchmark exists | `scripts/benchmark.py`. |
| One-command demo exists | `run_maple_shield_demo.py`, `tests/test_demo_runner.py`. |
| Demo is honest about unmeasured stages | `summary.json::stage_NOT_measured`, `tests/test_benchmark.py::test_benchmark_does_not_lie_about_yolo`. |
| Operator console and replay exist | `platform/operator-ui/`, `platform/docs/DEMO_QA.md`, platform tests. |
| CAIRN JSONL helper exists | `maple_shield_cairn_session.py`, `tests/test_cairn_session.py`. |

## Recommended External Positioning

Use this language:

> Maple Shield is a passive, edge-first drone detection and operator awareness
> platform. PR #13 adds a proof layer that measures CAIRN risk scoring on host
> CPU, emits audit-ready JSONL frame records, validates demo output, and maps
> product claims to repo evidence. Field performance, Jetson measurements,
> thermal/IR fusion, mesh coordination, Arctic validation, and STANAG
> conformance remain validation objectives for IC26.

Avoid this language until measured:

- "field-proven"
- "validated at -40 C"
- "30+ FPS on Jetson"
- "less than 50 ms detection-to-alert"
- "less than 2 percent false-positive rate"
- "STANAG-compatible" unless qualified as roadmap or scaffold
- "immune to electronic attack"
- "deployment-ready" unless clearly stated as a target

