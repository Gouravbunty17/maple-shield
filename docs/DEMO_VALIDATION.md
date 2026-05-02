# Maple Shield — Demo Validation

> What the one-command demo proves, and what it explicitly does **not**
> prove. If a question about Maple Shield maturity is downstream of a
> claim, the answer should be in this file or in
> `docs/DOC_REPO_ALIGNMENT.md`. If neither file backs the claim, the
> claim is roadmap and should be marked as such.

## TL;DR

```powershell
# from the repo root
python run_maple_shield_demo.py
```

The demo:

1. Runs the CAIRN risk-scoring stage 300 times against deterministic
   synthetic detections.
2. Writes `runs/demo-<UTC timestamp>/frames.jsonl` (one
   `CairnFrameRecord` per frame).
3. Writes `runs/demo-<UTC timestamp>/summary.json` (observed FPS,
   p50 / p95 / p99 of CAIRN scoring latency in ms, frame count,
   duration, stage-measured / stage-NOT-measured fields).
4. Writes `runs/demo-<UTC timestamp>/manifest.json` (Python version,
   platform, CAIRN engine version, argv).
5. Validates the JSONL by re-parsing every row.
6. Prints `RESULT: PASS` or `RESULT: FAIL` with concrete reasons.

Pass criteria:
- ≥ 60 FPS observed for the CAIRN scoring stage on host CPU.
- p95 of `cairn_scoring_ms` ≤ 50 ms on host CPU.
- All `frames.jsonl` rows are valid JSON and contain `schema`,
  `frame`, `ts`, `session_id`, `max_threat_level`, `risks`.
- `stage_measured` is `cairn_scoring` (not silently changed).

These thresholds are deliberately generous because the repo runs on
heterogeneous host CPUs. Tighter Jetson-specific thresholds will land
once we have real Jetson runs (roadmap).

## What this demo proves

| Claim | Backed by |
|---|---|
| CAIRN engine accepts detections, runs the risk engine, returns frame records | `frames.jsonl` rows produced and re-parsed in the demo |
| Audit-ready JSONL output exists | `frames.jsonl`; schema validated by `tests/test_benchmark.py::test_benchmark_jsonl_rows_are_valid_cairn_records` |
| `risk_score`, `threat_level`, `reasons`, `recommended_operator_action` populate per detection | `tests/test_cairn_session.py::test_session_writes_jsonl` |
| The CAIRN scoring stage runs ≥ 60 FPS on host CPU | `summary.json::frames_per_second_observed`; demo gates PASS on it |
| The CAIRN scoring stage p95 latency stays under 50 ms on host CPU | `summary.json::cairn_scoring_ms.p95`; demo gates PASS on it |
| Output is deterministic — same input, same record content (timestamps aside) | The synthetic detection generator in `scripts/benchmark.py::_build_synthetic_detections` is purely a function of `frame_idx`, `n_tracks`, `frame_w`, `frame_h` |

## What this demo does NOT prove

The repo must not assert any of these claims based on the demo alone.
They are listed in `summary.json::stage_NOT_measured` so a reader can
see the limits without reading code.

| Not measured by this demo | Why | Status |
|---|---|---|
| YOLO ONNX inference latency | The demo does not run a real model — synthetic detections only. | Roadmap. Run `maple_shield_mvp.py` with real weights to measure. |
| Camera capture latency | No camera. | Roadmap. |
| End-to-end detection-to-alert wall time on real video | No live source in the demo. | Roadmap. |
| Jetson Orin Nano performance | Demo runs on whatever host invoked it. | Roadmap. The same demo run on Jetson would produce a Jetson row. |
| <2% false-positive rate | No labelled-data evaluation in the demo. | Roadmap. |
| Arctic-condition validation | Synthetic frames, no field data. | Roadmap. `arctic_augment.py` exists for synthetic augmentation only. |
| Thermal / IR sensor fusion | Single-modality synthetic input. | Roadmap. Scaffolds in `cairn_edge/src/advanced/thermal_fusion.py`. |
| STANAG 4609 conformance | Demo produces JSONL, not STANAG. | Roadmap. Scaffold in `cairn_edge/src/advanced/stanag4609_export.py`. |
| Multi-node mesh | Single process. | Roadmap. Scaffold in `cairn_edge/src/advanced/mesh_sync.py`. |

## Reproducing the published demo numbers

```powershell
# from repo root
python run_maple_shield_demo.py --frames 300 --tracks 2

# inspect:
type runs\demo-<timestamp>\summary.json
```

The summary JSON is the single source of truth. Marketing material that
quotes a Maple Shield FPS or latency number must:

1. Cite a `summary.json` file produced by this demo or by
   `scripts/benchmark.py`.
2. Say what `stage_measured` was.
3. Say what `stage_NOT_measured` was.
4. Say what host the run was made on (`manifest.json::platform`).

Anything that quotes a number without those four facts is overclaiming.

## Test coverage

| Test | What it pins down |
|---|---|
| `tests/test_benchmark.py::test_benchmark_emits_frames_summary_and_manifest` | Files exist with documented contents. |
| `tests/test_benchmark.py::test_benchmark_jsonl_rows_are_valid_cairn_records` | Every row is a real `CairnFrameRecord`. |
| `tests/test_benchmark.py::test_benchmark_does_not_lie_about_yolo` | `stage_NOT_measured` includes YOLO, end-to-end, Jetson. |
| `tests/test_demo_runner.py::test_demo_runs_and_writes_outputs` | One-command demo PASSes and writes files. |
| `tests/test_demo_runner.py::test_demo_summary_shape_matches_documented_contract` | Summary contains every documented field. |
| `tests/test_cairn_session.py::test_session_writes_jsonl` | `CairnSession` JSONL output carries `risk_score`, `threat_level`, `reasons`, `recommended_operator_action`. |

If any of those tests starts failing, the doc is stale until the test
or the doc is fixed.
