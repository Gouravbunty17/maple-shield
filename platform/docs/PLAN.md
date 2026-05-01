# Maple Shield — Implementation Plan

Locked output of /autoplan. Implementation proceeds against this plan.

## Component map

```
                +-------------------+
   sample MP4   |    edge-agent     |   POST /detections
   or webcam -> | (Python+OpenCV)   | -----------------------+
                +-------------------+                        |
                                                             v
                                             +------------------------------+
                                             |       fusion-engine          |
                                             | track state + alert scoring  |
                                             +--------------+---------------+
                                                            |
                                                            | POST /alerts
                                                            v
                +-------------------+  HTTP    +--------------------------+
                |    operator-ui    | <======> |       command-api        |
                | React + Vite + TS |   WS     | FastAPI + SQLite + audit |
                +-------------------+          +--------------------------+
                                                            |
                                                            v
                                                    SQLite (incidents,
                                                    alerts, audit_log)
```

## Data contracts (pinned)

```jsonc
// Detection (edge-agent -> fusion-engine)
{
  "frame_id": "uuid",
  "ts": "2026-05-01T08:00:00.000Z",
  "camera_id": "cam-01",
  "detections": [
    { "class": "drone", "confidence": 0.91,
      "bbox": [x1, y1, x2, y2],
      "image_size": [w, h] }
  ]
}

// Track update (fusion-engine internal -> command-api)
{
  "track_id": "trk-001",
  "camera_id": "cam-01",
  "first_seen": "...", "last_seen": "...",
  "n_observations": 14,
  "smoothed_bbox": [...], "velocity_px_s": [vx, vy],
  "max_confidence": 0.94
}

// Alert (fusion-engine -> command-api)
{
  "alert_id": "uuid",
  "track_id": "trk-001",
  "severity": "high",     // info|low|med|high
  "rule": "dwell_over_threshold",
  "score": 0.82,
  "ts": "...",
  "snapshot_b64": "..."   // small JPEG
}

// Incident (command-api persisted)
{
  "incident_id": "inc-...",
  "status": "new",        // new|acknowledged|reviewed|closed
  "alerts": [alert_id, ...],
  "operator_notes": [],
  "created_ts": "...", "updated_ts": "..."
}

// Audit entry (append-only, hash chained)
{
  "seq": 42,
  "ts": "...",
  "operator_id": "op-123",
  "action": "incident.acknowledge",
  "target": "inc-abc",
  "payload": { ... },
  "prev_hash": "sha256:...",
  "hash":      "sha256:..."
}
```

## Build order (reduces risk by validating contracts early)

1. **command-api** first - it's the contract owner. Build models, schema,
   audit log, all endpoints with in-memory store, then SQLite.
2. **fusion-engine** second. Pure Python module + a thin HTTP shim. Easy to
   unit test in isolation (deterministic synthetic detections in/alerts out).
3. **edge-agent** third. Video reader + a deterministic mock detector that
   walks a hard-coded trajectory through the frame (so the demo is
   reproducible). Real OpenCV-backed detector is behind a `Detector`
   interface; we ship the mock and document how to swap.
4. **operator-ui** fourth. Vite scaffold + three views: Live, Alerts,
   Incident Replay. Talks to command-api over fetch + WebSocket.
5. **docs + scripts + tests** woven through, not at the end.

## Testing strategy

- `pytest` for each Python service. Target: track manager fixture-driven
  tests, alert scorer thresholds, audit chain verification, all API routes
  smoke-tested with httpx.
- `tests/test_compliance.py` greps the codebase for forbidden tokens
  (see `docs/COMPLIANCE.md`) and fails CI if any leak in.
- UI: `npm run build` must succeed; render-smoke test with a fixture API
  client. Full Playwright is out of MVP scope.
- One end-to-end test: feed sample synthetic detections through fusion ->
  command-api, GET incidents, assert one alert with severity >= med.

## Run / dev story

```bash
# install
make setup        # python deps + npm i
# run all (one terminal)
make dev          # spawns command-api, fusion-engine, edge-agent (mock), and Vite dev server
# tests
make test         # pytest -q + ui build
```

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Real detector swap breaks the contract | `Detector` ABC; mock and any real impl conform. Contract test in edge-agent. |
| Audit chain bug invalidates trust | Verifier runs on every read of the log; UI surfaces the result. |
| Compliance drift over time | Forbidden-token test runs in CI on every change; the test list is in COMPLIANCE.md. |
| OS path issues (Windows host / Linux sandbox) | Build & test in Linux sandbox, deliver a copy in Cowork outputs. README documents both paths. |

## Plan coherence check

- Boundaries from SCOPE.md are reflected in COMPLIANCE.md and enforced by
  `tests/test_compliance.py`. ✓
- Every MVP success criterion has at least one task and one test. ✓
- No component requires a forbidden capability. ✓
- Build order respects data flow direction (consumers before producers
  where contracts matter). ✓

Plan is coherent. Proceeding to implementation.
