# Maple Shield — Test Plan

## Layers

| Layer | Tool | Where |
|-------|------|-------|
| Unit | pytest | `*/tests/` per service |
| Contract | httpx + TestClient | `command-api/tests/test_api.py`; covers request/response shapes |
| End-to-end | scripted detection feed | `tests/test_e2e.py` |
| Compliance | regex grep | `tests/test_compliance.py` |
| UI build | tsc + vite | `npm run build` (in CI) |

## What each layer proves

- **Unit / store**: SQLite schema, audit chain hash, status lifecycle,
  track lifecycle, scoring ladder.
- **Contract**: every documented endpoint round-trips a real request.
- **End-to-end**: synthetic detections enter fusion → fusion produces alerts
  → command-api creates an incident → audit log records it → replay returns
  the timeline.
- **Compliance**: the codebase contains none of the forbidden tokens listed
  in `docs/COMPLIANCE.md`.
- **UI**: the bundle compiles cleanly with strict TypeScript.

## CI commands

```bash
make test-compliance   # forbidden-token grep
make test-py           # pytest
make build-ui          # tsc + vite build
make test              # all of the above
```

## Manual smoke test

```bash
make setup
make dev               # all four services
# in another terminal:
curl http://localhost:8080/healthz
curl http://localhost:8080/alerts
# open http://localhost:5173
```

Expect: alerts populating in the UI as the mock detector walks across the
synthetic frame, an incident auto-opening when severity reaches `med`,
status lifecycle works in the Replay view, audit chain shows ✓ verified.
