# Maple Shield — /ship report

## Summary

Built the Maple Shield platform MVP (passive airspace monitoring) inside
the existing `maplesilicon-site` repository under `maple-shield/platform/`,
preserving the marketing page (`maple-shield/index.html`) and all
uncommitted local changes.

## Deliverable

`maple-shield/platform/` — 50 files, four services, full doc set, 34 tests.

```
platform/
├── README.md, Makefile, requirements.txt, pytest.ini, conftest.py, .gitignore
├── command-api/          # FastAPI + SQLite + WebSocket; owns contracts
│   ├── app/              # main.py, models.py, store.py, routers/
│   └── tests/            # 16 unit + contract tests
├── fusion/               # tracker + scorer (importable package)
├── fusion-engine/        # holds tests for fusion package
│   └── tests/            # 10 tests
├── edge-agent/           # video ingest + Cairn-style mock detector
│   ├── edge_agent/       # detector.py, source.py, main.py
│   └── tests/            # 4 tests
├── operator-ui/          # React + Vite + TypeScript SPA
│   ├── src/              # 4 views + lib/api.ts + types.ts + styles.css
│   └── package.json + tsconfig.json + vite.config.ts
├── docs/                 # SCOPE, COMPLIANCE, ARCHITECTURE, PLAN, TEST_PLAN, RUNBOOK, REVIEW, SHIP_REPORT
├── tests/                # cross-cutting compliance + e2e (4 tests)
├── samples/, scripts/    # dev script, sample placeholder
```

## Tests run

| Suite | Result |
|-------|--------|
| `pytest -q` (root) | **34 passed** |
| compliance grep (forbidden tokens, DELETE endpoints) | **clean** |
| `npm run build` (operator-ui) | **clean**, 175 KB JS bundle (56 KB gzipped) |
| End-to-end live driver (`/tmp/qa_driver.py`) | **OK** — 30 frames fed; 3 alerts (low/med/high), 1 incident with 2 alerts (consolidation works), audit chain verified across 6 actions |

## What works

- **edge-agent** ingests frames (mock or video file), runs `CairnMockDetector`, posts `DetectionFrame` to fusion. Detector is restricted to `drone` class only — enforced at the dataclass level and again defensively at the tracker.
- **fusion-engine** maintains tracks (greedy IoU + EMA bbox + velocity), retires after `max_misses`, scores three rules (`single_obs` / `dwell_over_threshold` / `persistent_high_confidence`) into severity ladder, debounces alerts per-track-rule, GCs dedup state when tracks retire.
- **command-api**: full REST surface (`/healthz`, `/readyz`, `/alerts`, `/incidents` with status lifecycle and notes, `/export`, `/replay`, `/audit`) plus `WS /ws/alerts` for live UI streaming. Auto-incidents consolidate per track. SQLite-backed; in-memory by default.
- **operator-ui**: four views — Live, Alerts (filtered), Replay (status lifecycle, notes, exportable summary), Audit (chain verification badge). Vite proxy to API on `:8080`.
- **Audit log**: hash-chained, append-only, no DELETE endpoints anywhere. The verifier runs on every `/audit` read and the UI badges the result; tampering test confirms detection of a manual SQL edit.
- **Compliance**: forbidden-token grep + DELETE-endpoint grep + a self-test that the negation regex isn't too permissive. Lawful-use ack via env var, surfaced as a banner in the UI when unset.

## What remains

- **Real frame source** for Live view bbox overlay. The state plumbing is there; what's missing is a frame transport (MJPEG/JSMPEG/WebRTC) — this is integrator work, not platform work.
- **Real detector**. Swap `CairnMockDetector` for a CAIRN-conformant implementation that satisfies the `Detector` Protocol. Class set must remain `{drone}`.
- **AuthN/Z**. Out of MVP — `operator_id` is currently passed in as a body field; in production this comes from the SSO/proxy layer.
- **Multi-camera UI grid**. The API is camera-aware; the UI currently displays a single feed.
- **Playwright UI tests**. `npm run build` is clean; full browser-render tests are next.
- **Production deployment** (TLS, retention sweeper, log shipping) — left to the integrator.

## Files changed (in user's repo)

- **Added** (untracked): `maple-shield/platform/` — 50 files.
- **Not touched**: existing marketing page (`maple-shield/index.html`), logo assets, all other top-level files. Pre-existing dirty changes (CNAME, about/index.html, index.html, etc.) are left exactly as you had them.
- **Stray artifact**: `maple-shield/.platform_writetest` (0-byte; from a write-permission probe early in the session — Cowork's Linux sandbox can't unlink files on the Windows mount). Safe to delete via Windows Explorer.
- **No git operations were run** in your repo. Run `git add maple-shield/platform/ && git commit -m "..."` yourself when ready.

## Compliance assumptions / boundaries

These are immutable product constraints, not deferred TODOs:

1. **Passive observation only.** No RF transmit, no jamming, no neutralization, no interception, no take-over, no targeting, no weapons, no autonomous engagement, no instructions for defeating drones.
2. **Class set.** Only `drone` is detected. Person/face/vehicle classes are explicitly excluded by detector contract and dropped defensively in the tracker.
3. **Append-only.** No DELETE endpoint exists anywhere; the test suite would fail if one were added. Status changes replace deletes.
4. **Audit chain.** Hash-chained, verified on every read, surfaced in the UI. Treat a chain break as a potential compromise.
5. **Lawful-use ack.** Deploying organisations must set `MAPLE_SHIELD_LAWFUL_USE_ACK=true`; the UI warns until they do. The platform does not validate jurisdictional rules itself — that's an organisational obligation.
6. **PII minimisation.** Operator names are stored in audit/notes; export bundles include them by default. A redaction step is a known future addition for cross-organisation sharing.
7. **No facial recognition** of any kind, full stop.
8. **Decision support.** Alerts are advisory. The platform never originates state changes against a target. All operator actions are human-initiated and logged.

## Process notes

- gstack skills cannot be loaded as slash commands in Cowork; the `~/.claude/skills/gstack` folder is a Cowork-protected path. The workflow steps (`/health`, `/office-hours`, `/autoplan`, `/review`, `/qa`, `/ship`) were therefore *improvised* — each step performed and labelled, but procedural fidelity to the actual gstack `SKILL.md` files could not be verified.
- The user's GitHub repo at `github.com/Gouravbunty17/maple-shield` was referenced but not fetched or modified; this work lives in the local `maplesilicon-site` clone under `maple-shield/platform/`.
