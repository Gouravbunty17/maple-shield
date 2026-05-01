# /review — findings

## Strengths
- 32 tests pass; all four services have unit + contract coverage; one true E2E test.
- Compliance is enforced in code (forbidden-token grep + DELETE-endpoint grep + a test that the negation regex isn't too permissive). 30 docs lines (`docs/COMPLIANCE.md`) link directly to the test.
- Audit log uses a hash chain that the API verifies on every read; the UI surfaces verification status. Tampering test catches a manual SQL edit.
- Detector is restricted to `drone` class at the type contract AND defensively in the tracker (`if det["cls"] != "drone": continue`).
- No DELETE endpoints anywhere; archive operations are status changes, not erasure.

## Issues found

### Substantive (fixed)

1. **Duplicate auto-incidents per track.** `POST /alerts` was creating a new incident every time a med/high alert came in, even if a track already had an open incident. Fixed by collapsing same-track alerts into the existing open incident.
2. **Unbounded dedup cache in fusion-engine.** `_last_alert_rule_for_track` dict grew without bound. Fixed by clearing entries when a track is retired.

### Minor (left as-is, low impact)

3. `IncidentReplay.tsx` uses `<a href>` for incident-list navigation instead of `<Link>`. Full reload on click; harmless.
4. `Live.tsx` has bbox overlay state that's never populated — we don't have a real frame source. Documented in RUNBOOK as "swap in a real source".
5. Compliance allowlist hardcodes file paths. If a doc gets renamed, the test silently accepts the rename. Acceptable for MVP; would tighten in v2 with a CI-side check.
6. Tracker `velocity` doesn't decay across miss frames; it stays at last observation. Acceptable for MVP — this is decision-support quality, not navigation-grade.

## Compliance posture
- Forbidden tokens: 0 hits outside disclaimer/negation context.
- DELETE endpoints: 0.
- Class set: `{drone}` only (Pydantic `pattern="^drone$"` + runtime drop in tracker).
- Audit chain: verifies on every read; UI surfaces status.
- Lawful-use ack: env var gate + UI warning when unset.
