# Maple Shield — Architecture

Maple Shield is a four-process system, deliberately split so each component
has a single responsibility and a clear contract with the next.

```
camera/file -> edge-agent -> fusion-engine -> command-api -> operator-ui
                 (detect)     (track+score)    (persist)      (render)
```

## edge-agent
- **Responsibility:** ingest video, run a detector, emit DetectionFrame packets.
- **Interface in:** OpenCV `VideoCapture` source or the bundled `MockSource`.
- **Interface out:** HTTP `POST /detections` to fusion-engine.
- **Detector contract:** the `Detector` Protocol in `edge-agent/src/detector.py`.
  The MVP ships `CairnMockDetector`; production deployments swap in a real
  detector that conforms to the same Protocol.
- **Class set:** restricted to `{drone}` only — the platform does not produce
  person, face, or vehicle detections by design.

## fusion-engine
- **Responsibility:** maintain track state across frames; score tracks into
  alerts.
- **Tracker:** greedy IoU association + exponential bbox smoothing, retire
  tracks after `max_misses` frames without an observation.
- **Scorer:** rule-based ladder
  - `single_obs` (low) — one confident detection
  - `dwell_over_threshold` (med) — sustained presence
  - `persistent_high_confidence` (high) — long, confident track
- **Output:** `POST /alerts` to command-api. Debounced so the same rule
  doesn't fire twice on the same track.
- **No engagement:** the fusion-engine never emits an action against a
  target. It writes alerts and exposes track state for read.

## command-api
- **Responsibility:** persistence + API for the UI.
- **Storage:** SQLite (default in-memory; set `MAPLE_SHIELD_DB=/path/to.db`).
- **Endpoints:**
  - `GET /healthz`, `GET /readyz`
  - `POST /alerts` (from fusion), `GET /alerts`, `GET /alerts/{id}`
  - `GET /incidents`, `GET /incidents/{id}`, `POST /incidents`,
    `PATCH /incidents/{id}/status`, `POST /incidents/{id}/notes`,
    `GET /incidents/{id}/export`
  - `GET /replay/{incident_id}`
  - `GET /audit`
  - `WS  /ws/alerts` (live alert stream)
- **Auto-incident:** alerts with severity ≥ `med` automatically open a new
  incident as a bookkeeping convenience. The action is logged to the audit
  log; it does not initiate any state change against any target.

## operator-ui
- **Responsibility:** human interface.
- **Views:** Live (recent alerts + bbox overlay placeholder), Alerts (filtered
  list, pivot to replay), Replay (per-incident timeline, status lifecycle,
  notes, exportable summary), Audit (chain verification view).
- **Auth:** out of MVP. Production deployments wire identity/SSO at the
  edge proxy and pass `operator_id` through.

## Audit log

Every state-changing action goes through command-api and lands in the
`audit_log` table as a hash-chained record:

```
hash_n = SHA256( seq | ts | operator | action | target | payload | hash_(n-1) )
```

`GET /audit` recomputes the chain on every read; the UI surfaces the
verification result. The store has no `delete` method; archive operations
are status changes, not deletions.

## Threat model (high level)

| Threat | Mitigation |
|--------|-----------|
| Tampered audit log | Hash chain verified on every read. |
| Operator denying an action | Audit chain is the record of truth. |
| Misconfigured detector emits non-drone classes | Tracker drops non-`drone` detections defensively. |
| Compliance drift in code | `tests/test_compliance.py` greps for forbidden tokens; CI fails on hit. |
| Deployment in unlawful zone | `MAPLE_SHIELD_LAWFUL_USE_ACK` env var; UI surfaces a warning when unset. |
| PII leak via export | Export bundles do not include operator names beyond the audit log; redaction step is left to the integrator. |
