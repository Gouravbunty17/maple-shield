# Maple Shield — Runbook

## First run

```bash
make setup
export MAPLE_SHIELD_LAWFUL_USE_ACK=true   # confirm lawful deployment
make dev
```

Open http://localhost:5173 (operator-ui), API at http://localhost:8080.

## Replacing the mock detector

The mock detector in `edge-agent/src/detector.py` exists so the demo runs
without any model weights. To wire in a real detector (CAIRN or otherwise):

1. Implement `Detector.detect(frame_idx, frame_w, frame_h) -> List[Detection]`.
2. Restrict the class set to `{drone}`. Anything else MUST be filtered before
   the call returns.
3. Swap the constructor in `edge-agent/src/main.py`.
4. Re-run `make test`.

The detector contract is intentionally narrow: a real implementation can use
any model architecture, on CPU or accelerator, as long as it returns drone
detections only.

## Backups and retention

- SQLite file location: `MAPLE_SHIELD_DB` env var (default `:memory:`).
- Default retention policy: long-term retention is per-incident only;
  short-term frame retention is bounded by free space.
- Audit log: append-only. Archive by copying the database file off-host;
  do not edit it in place.

## What to do when the audit chain fails verification

1. The UI shows `✗ chain broken at seq N`.
2. The break is at the first row where `prev_hash` no longer matches the
   previous row's `hash`, or where the row's own `hash` does not match its
   recomputed value.
3. Treat the database as compromised from row N onward. Rotate to a new
   database; preserve the old one as evidence; investigate the host.

## What this platform does NOT do

If you are looking for any of the following, you are in the wrong product:

- Drone interception, jamming, neutralization, take-over, or damage
- Targeting or weapon-control workflows
- Autonomous engagement
- Facial recognition / biometric identification

These are out of scope by product design and enforced by tests.
