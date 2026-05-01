# Maple Shield — Compliance Boundaries

This document is **normative**. The platform must enforce these boundaries in
code; the test suite includes checks that fail if any forbidden surface is
introduced.

## Hard boundaries

| # | Boundary | Enforcement |
|---|----------|-------------|
| 1 | Passive observation only - no active RF emissions toward targets | No RF transmitter driver code. No "transmit" verb in any API. Test: `tests/test_compliance.py::test_no_forbidden_symbols` greps the codebase for forbidden tokens. |
| 2 | No interception / neutralization / jamming | No effector, weapon, jammer, interceptor, or "engage" interfaces. Same grep test as (1). |
| 3 | No autonomous engagement | The platform never originates a state change against a target. The only "actions" the platform takes are: write logs, raise alerts to the UI, send notifications. Operators are the only entity that can change incident state. |
| 4 | No facial recognition | Detector class set is restricted to `{drone}` (and explicitly excludes `person`, `face`, `vehicle`). The fusion engine drops detections for unsupported classes and logs them as a config error. |
| 5 | Auditability | Every operator action goes through `command-api` and is appended to a hash-chained audit log. Logs are append-only; the API has no delete endpoint. |
| 6 | Data minimisation defaults | Default frame retention is short (`24h`); long-term retention is per-incident only. PII fields (operator name, email) are kept separately and can be redacted on export. |
| 7 | Lawful deployment | The platform is opinionated about *not* shipping - the README and `/healthz` warn if deployed without an explicit `MAPLE_SHIELD_LAWFUL_USE_ACK=true` environment variable, signalling the deploying org has reviewed jurisdictional rules. |

## Forbidden tokens

The following tokens MUST NOT appear in source code outside this document
and the test that enforces the rule. The test fails the build if they do.

```
jam, jammer, jamming, neutralize, neutralise, neutralization,
interceptor, intercept_target, kinetic, weapon, weaponize, engage_target,
spoof_gps, gps_spoof, hijack, takeover_drone, kill_switch, fire_command
```

## Data classes the platform handles

| Class | Examples | Handling |
|-------|----------|----------|
| Detection metadata | bbox, class, confidence, timestamp | Retained per incident; sample retention 24h. |
| Frame stills | JPEG snapshot at alert time | Retained per incident only, encrypted-at-rest in production. |
| Track state | Kalman state, velocity | In-memory, derived; not persisted long term. |
| Audit log | Operator id, action, timestamp, prev_hash, hash | Append-only, retained per regulatory minimum. |
| Operator PII | Name, email | Stored separately, redactable. |

## Operator obligations (surfaced in UI)

- Confirm lawful monitoring zone before enabling a camera.
- Acknowledge data retention policy on first login.
- Treat alerts as decision support, not as authorisation to act.
