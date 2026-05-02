# Maple Shield platform — v0.1.0

> Initial MVP of the Maple Shield operator platform. Lawful, passive
> airspace monitoring. No interception, no jamming, no neutralization,
> no targeting, no autonomous engagement.

## What's in this release

Four services and an operator UI under `platform/`:

- **edge-agent** — video ingest + detector contract; ships a deterministic
  mock detector restricted to the `drone` class.
- **fusion-engine** — IoU tracker + alert scorer with a three-rule severity
  ladder (`single_obs` / `dwell_over_threshold` / `persistent_high_confidence`).
- **command-api** — FastAPI + SQLite. Endpoints for alerts, incidents,
  notes, status lifecycle, replay, exportable summary, hash-chained audit
  log, health, and a WebSocket for live alerts.
- **operator-ui** — React + Vite + TypeScript. Live, Alerts, Replay (with
  status lifecycle, notes, exportable Markdown summary), Audit (chain
  verification badge).
- **docs** — SCOPE, COMPLIANCE, ARCHITECTURE, PLAN, TEST_PLAN, RUNBOOK,
  REVIEW, SHIP_REPORT, PHASE2_PLAN.

## Boundaries

Hard product constraints, enforced in code by `platform/tests/test_compliance.py`:

- Passive observation only.
- No interception, jamming, neutralization, take-over, spoofing.
- No targeting, weapons, or weapon-control workflows.
- No autonomous engagement.
- No facial recognition.
- No DELETE endpoints anywhere; archive is a status change.
- Class set restricted to `drone`.

## Verification

- `pytest -q` from `platform/`: **34 passed**
- `npm run build` (operator-ui): **clean**, 175.79 KB JS / 56.90 KB gzipped
- `npm audit` (full and `--omit=dev`): **0 vulnerabilities**
- Live end-to-end driver: 30 frames → 3 severities → 1 incident with 2
  alerts (consolidation works) → operator lifecycle → markdown export →
  audit chain verified across 6 actions.

## Known gaps (addressed in phase 2)

- Mock detector to be replaced by an adapter to the existing
  `cairn_engine` package.
- Live frame transport (MJPEG/WebRTC) is integrator work; bbox overlay
  state is plumbed but currently empty.
- AuthN/Z, multi-camera grid, Playwright UI tests, production hardening
  remain on the v1.x roadmap.

## Compliance assumptions surfaced for the deploying organisation

1. Camera coverage is in a lawful monitoring zone for your jurisdiction.
2. Operators are trained and authorised under your local rules.
3. Set `MAPLE_SHIELD_LAWFUL_USE_ACK=true` only after that review.
4. The platform produces decision support; humans are the only entities
   that can change incident state.

---

# Maple Shield platform — v0.2.0 (CAIRN adapter)

> Phase 2: integrate the platform with the existing `cairn_engine` package.
> Boundaries unchanged. Additive integration inside `platform/`; no existing
> root CAIRN engine, edge package, or `maple_shield_*.py` scripts modified.

## What's added

- **`platform/edge-agent/edge_agent/cairn_adapter.py`** — `CairnSourceDetector`
  implements the platform `Detector` Protocol against `CairnEngine.process_frame()`.
  Drone-only at the adapter (defensive); CAIRN threat surfaced only via
  `Detection.raw` display hint (severity ownership stays in fusion).
  Pinned compatibility check warns on minor-version drift (`CairnVersionWarning`).
- **`platform/edge-agent/edge_agent/source_cairn.py`** — `CairnFrameSource`
  yields `(frame_idx, frame)` from a webcam index or video file path.
- **`platform/command-api/app/routers/cairn_health.py`** — `GET /cairn/health`
  returns engine name, version, package_version, compatible flag,
  frames_processed, runtime_s, started_ts, risk_config.
- **`platform/operator-ui/src/views/CairnHealth.tsx`** — pill-style
  status badge slotted into the Live view (`role="status"`, `aria-expanded`).
  Amber state on engine-unreachable, accent-cyan when healthy.
- **Tracker tweak**: `platform/fusion/tracker.py` now honours an external
  `track_id` from CAIRN verbatim before running IoU matching, so identity
  is preserved across the boundary. Refactored shared per-track update
  into `_apply_detection`. Also propagated through `fusion/main.py` and
  `edge-agent/main.py`.

## Verification

- `pytest -q` from `platform/`: **40 passed** (+6 over phase 1)
- `pytest -q tests/` (repo root): **27 passed** (unchanged)
- Live e2e with `track_id="cairn-trk-9"` injected: external id preserved,
  3 alerts, 1 incident, audit chain verified, `/cairn/health` reports
  compatible.

## Boundaries reaffirmed

No interception, jamming, neutralization, take-over, targeting, weapons,
autonomous engagement, facial recognition, or DELETE endpoints. Platform
class set still restricted to `drone`. Compliance grep (whole repo): 0
non-disclaimer hits.
