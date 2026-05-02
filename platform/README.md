# Maple Shield

> **Edge AI Platform for Real-Time Drone Detection & Airspace Security**
> *Protecting Airspace with Intelligence.*

Maple Shield is a lawful, **passive** airspace monitoring platform for trained
operators. It detects drones in monitored airspace, tracks them over time,
raises scored alerts, persists incidents with hash-chained audit trails,
and provides a web UI for live monitoring and incident review.

The platform is observational. It does **not** intercept, jam, neutralize,
target, or otherwise interfere with drones. See `docs/SCOPE.md` and
`docs/COMPLIANCE.md` for hard product boundaries.

## Components

| Component | Purpose | Stack |
|-----------|---------|-------|
| `edge-agent/` | Video ingest + detection metadata producer | Python 3.10, OpenCV, httpx |
| `fusion-engine/` | Track state + alert scoring | Python 3.10, NumPy |
| `command-api/` | Incidents, tracks, alerts, replay, audit log, health | FastAPI + SQLite |
| `operator-ui/` | Live monitor, alert list, incident replay | React + Vite + TypeScript |
| `docs/` | Architecture, scope, compliance, plan, test plans | Markdown |

## Quickstart

```bash
make setup          # python deps + npm i
make dev            # all services + Vite dev server
make demo           # local demo launcher with mock fallback
make test           # pytest + UI build + compliance grep
```

Open http://localhost:5173 (operator-ui), the API is at http://localhost:8080.

A synthetic sample feed is shipped under `samples/`. The default detector is a
deterministic mock that reproduces a stable trajectory across the frame so
the demo is repeatable.

For a guided demo flow, see `docs/DEMO.md`.

Phase 2 adds `edge_agent.cairn_adapter.CairnSourceDetector`, which bridges the
repo's existing `cairn_engine` package into the same detector contract. The
mock remains the CI default; CAIRN integration details live in
`docs/CAIRN_INTEGRATION.md`.

## Lawful use

Set `MAPLE_SHIELD_LAWFUL_USE_ACK=true` in the deployment environment after
your organisation has confirmed the deployment is lawful in your jurisdiction.
The platform's `/healthz` endpoint emits a warning until this is set.

## Safety boundaries (immutable)

- No interception, neutralization, jamming, spoofing, or take-over.
- No targeting, weapons, or weapon-control workflows.
- No autonomous engagement.
- No facial recognition.
- All operator actions are logged in a hash-chained audit log.

A test (`tests/test_compliance.py`) fails the build if any forbidden capability
is introduced.

## License

Internal MVP scaffold. Add your organisation's license before distribution.
