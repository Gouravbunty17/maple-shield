# Maple Shield — NATO IC26 Document ↔ Repo Alignment

> **Purpose.** Map every claim in the NATO IC26 document to repo evidence,
> with an honest status label. The document can stay ambitious; the repo
> stays defensible. Where a claim is not yet supported in the repo we say
> so out loud.

Companion artifacts:

- `docs/NATO_IC26_REPO_ALIGNED_RESPONSE.md` is a clean repo-aligned rewrite
  of the NATO IC26 response.
- `docs/NATO_IC26_CLAIM_ALIGNMENT_MATRIX.md` maps high-risk source-PDF claims
  to repo-backed replacement language.

## Status legend

| Label | Meaning |
|---|---|
| **proven** | Repo contains running code AND a reproducible measurement / test that backs the claim. |
| **demo-supported** | Repo contains running code that exercises the claim against deterministic / synthetic input. Real-world measurement is still pending. |
| **in progress** | Code exists but the claim is not yet exercisable end-to-end. |
| **roadmap** | Claim is intentionally out of scope right now. No code, no fake stub. |

---

## Alignment table

| # | NATO doc claim | Status | Evidence in repo | Honest gap |
|---|---|---|---|---|
| 1 | Passive vision-first C-UAS early warning | **proven** | `platform/docs/COMPLIANCE.md` enumerates the boundary; `platform/tests/test_compliance.py` greps for forbidden tokens (jam / neutralize / intercept / weapon / engage / spoof / hijack / kill switch / fire command) and fails CI on any non-disclaimer hit. The whole platform tree has no DELETE endpoints anywhere. | None. The boundary is enforced in code. |
| 2 | CAIRN-powered detection / risk engine | **demo-supported** | `cairn_engine/` package implements `CairnEngine.process_frame()`, `CairnRiskEngine`, `CairnDetection`, `CairnRiskResult`, `CairnFrameRecord`. `cairn_demo.py` runs one synthetic frame end-to-end. `platform/edge-agent/edge_agent/cairn_adapter.py` bridges CAIRN into the platform `Detector` Protocol with drone-only enforcement. | The risk-scoring layer is real and tested. The detector layer (real ONNX inference) is wired in `maple_shield_mvp.py` but not yet routed through `CairnEngine` end-to-end on a live frame source. Phase 3 closes that loop. |
| 3 | Edge-first / no cloud dependency | **demo-supported** | All packages import locally only — no cloud SDKs, no remote API calls in `cairn_engine/`, `cairn_edge/`, `platform/`. Local SQLite for incidents. | We do not yet ship a hardened "no network" build flag. The repo *is* offline-capable today; we just haven't asserted it programmatically. Roadmap. |
| 4 | Jetson Orin Nano target | **in progress** | `cairn_edge/scripts/jetson_profile_check.py` exists. `cairn_edge/configs/edge.node.yaml` has Jetson-flavored knobs. | Not yet measured on real Jetson hardware. We do not claim 30 FPS *on Jetson* until we have a Jetson run. Roadmap label below. |
| 5 | 30+ FPS target | **demo-supported (host CPU)** | `scripts/benchmark.py` (this PR) measures FPS of the CAIRN risk-scoring stage end-to-end with deterministic synthetic input on host CPU. The number we publish is host-CPU-only and labelled as such. | We **do not** claim 30 FPS on Jetson, and we **do not** claim 30 FPS for the YOLO inference stage. Both are roadmap. |
| 6 | <50 ms detection-to-alert target | **demo-supported (CAIRN scoring stage only)** | `scripts/benchmark.py` measures p50/p95/p99 latency of `CairnEngine.process_frame()` per call. This is the **CAIRN risk-scoring** latency, not full YOLO-to-alert. | The full YOLO-inference-to-alert path on real video is not measured here. We do not claim a sub-50ms full pipeline number. We label what we measure. |
| 7 | JSON / audit-ready records | **proven** | `cairn_engine.schemas.CairnFrameRecord.to_json()` produces `{schema, frame, ts, iso_time, session_id, frame_w/h, fps, infer_ms, max_threat_level, max_risk_score, risks[…], health}`. `scripts/benchmark.py` emits one such record per frame as JSONL plus a `summary.json`. `platform/command-api` keeps a hash-chained audit log; `platform/tests/test_compliance.py::test_no_delete_endpoint_anywhere` enforces append-only. | None for the schema. |
| 8 | Operator console / replay | **proven** | `platform/operator-ui/` (React + Vite + TS) ships Live, Alerts, Replay (with status lifecycle, notes, exportable summary), Audit views. Repo and platform tests are green. `npm run build` clean (176.53 KB JS / 57.15 KB gz). 0 npm vulns. | None. |
| 9 | C2-friendly structured outputs | **demo-supported** | `cairn_edge/src/c2_emitters.py` exists. CairnFrameRecord JSONL is C2-friendly. `maple_shield_cot.py` emits CoT XML. | We do not claim STANAG conformance. STANAG = roadmap. The CoT path exists but is not exercised by the repeatable demo yet. |
| 10 | Future multi-node mesh | **roadmap** | `cairn_edge/src/advanced/mesh_sync.py` exists as a scaffold. | Not exercised by any demo or test in this PR. Listed in `README.md` Roadmap. |
| 11 | Future visible + thermal/IR fusion | **roadmap** | `cairn_edge/src/advanced/thermal_fusion.py` and `configs/thermal.yaml` exist as scaffolds. | Not exercised by any demo or test. Listed in `README.md` Roadmap. |

## Things we explicitly do NOT claim

These appear in the IC26 doc as ambitions / future direction. The repo
**must not** assert them as proven until measurement exists:

- **STANAG 4609 conformance** — scaffold exists in `cairn_edge/src/advanced/stanag4609_export.py` but is not part of the verified demo path. Roadmap.
- **Thermal / IR fusion validation** — scaffold only. Roadmap.
- **Multi-node mesh validation** — scaffold only. Roadmap.
- **Arctic-condition validation** — `arctic_augment.py` exists for synthetic data augmentation; no real-world Arctic deployment data. Roadmap.
- **<2 % false-positive rate** — no field test data in repo. Roadmap.
- **<50 ms full detection-to-alert latency on Jetson** — not measured. Roadmap. The benchmark in this PR measures the **CAIRN scoring stage on host CPU**, and labels it that way.
- **Jetson power-budget validation** — no measured numbers. Roadmap.

## What this PR adds (proof layer)

1. **`scripts/benchmark.py`** — repeatable measurement: FPS, per-frame
   wall-time latency (p50/p95/p99), frame count, run duration.
   Produces `runs/<session>/frames.jsonl` and `runs/<session>/summary.json`.
   Default mode uses deterministic synthetic detections so the run is
   reproducible without a camera.
2. **`run_maple_shield_demo.py`** — one-command wrapper that calls the
   benchmark, verifies output integrity, and prints PASS / FAIL with
   concrete numbers. Logs land under `runs/demo-<timestamp>/`.
3. **`maple_shield_cairn_session.py`** — small importable helper that any
   caller (including a future `mvp.py` `--cairn-jsonl` flag) can use to
   open a CairnEngine session, push detections in, and write
   CairnFrameRecord JSONL out.
4. **`tests/test_benchmark.py` and `tests/test_demo_runner.py`** — pytest
   covers that benchmark and demo emit valid JSONL + summary.json with
   the contract above.
5. **`docs/DEMO_VALIDATION.md`** — what the demo proves and what it does
   not. Reproducible commands. Pass / fail criteria.
6. This file (`docs/DOC_REPO_ALIGNMENT.md`).
7. Updated `README.md` with explicit "Verified today" / "In progress" /
   "Roadmap" sections.

## Reproducible commands

```powershell
# from repo root
python -m pytest -q tests
cd platform
python -m pytest -q
cd ..
python run_maple_shield_demo.py
# inspect: runs/demo-<timestamp>/summary.json, runs/demo-<timestamp>/frames.jsonl
```

## Maintenance rule

Any new claim added to `README.md` or marketing material must be backed by
a row in this alignment table with a status of **proven** or
**demo-supported**, plus a pointer to the file and test that support it.
Anything else goes in **roadmap** until the evidence lands.
