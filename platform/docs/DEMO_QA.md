# Maple Shield Demo QA

Date: 2026-05-02

Branch tested: `codex/phase3-proof-layer`

## Scope

This QA pass verified the browser-visible operator console against a live local
demo stack. The test used the deterministic mock edge feed, so it required no
camera, video file, model weights, or cloud service.

## Stack

| Service | URL |
| --- | --- |
| command-api | `http://localhost:8080` |
| fusion-engine | `http://localhost:8090` |
| operator-ui | `http://127.0.0.1:5173` |

Use `127.0.0.1` for the UI when multiple local Vite apps are running. In this
session, another app was listening on the IPv6 `localhost` route.

## Data Seed

The edge-agent mock feed ran for 80 frames at 120 FPS:

```bash
python -m edge_agent.main --source mock --detector mock --n-frames 80 --fps 120 --fusion http://localhost:8090
```

API evidence after the feed completed:

| Check | Result |
| --- | --- |
| Alerts | 19 |
| Incidents | 5 |
| Audit chain | verified |

## Browser Evidence

Headless Chrome rendered each route and saved local screenshots under
`platform/.demo-logs/browser-qa/screens/`.

| Route | Evidence |
| --- | --- |
| `/` | Live monitor rendered with CAIRN status and recent alerts |
| `/alerts` | Alerts table rendered with severity filters and review links |
| `/replay` | Incident replay list rendered with five incidents |
| `/replay/inc-4341467181` | Incident detail rendered with timeline and action buttons |
| `/audit` | Audit log rendered with chain verified status |

Screenshots generated locally:

```text
platform/.demo-logs/browser-qa/screens/live.png
platform/.demo-logs/browser-qa/screens/alerts.png
platform/.demo-logs/browser-qa/screens/replay.png
platform/.demo-logs/browser-qa/screens/replay-detail.png
platform/.demo-logs/browser-qa/screens/audit.png
```

## Result

Browser-visible demo QA passed. The operator console rendered the expected
views, the seeded data appeared in the UI, and the audit page reported a
verified chain.

## Follow-Up

The next useful upgrade is to automate this browser route check in a reusable
script. Keep it optional and local-only so CI does not need a browser runtime.
