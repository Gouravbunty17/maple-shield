# Maple Shield — Scope (locked)

> **Tagline:** Protecting Airspace with Intelligence
> **Positioning:** Edge AI Platform for Real-Time Drone Detection & Airspace Security
> **Status:** Scope locked during /office-hours. Any change to scope requires a new scope review and an entry in `docs/CHANGELOG.md`.

## Mission

Build a lawful, **passive** airspace monitoring platform that helps trained operators
**detect, track, alert on, log, and review** drone activity in monitored airspace.

The platform exists to give operators awareness so they can follow their
organisation's policies, escalate to appropriate authorities, and produce
auditable records. The platform itself is observational.

## In scope (MVP)

1. **Ingest** a sample video file *or* a single live camera stream and produce
   per-frame detection metadata (class, bounding box, confidence, timestamp).
2. **Track** detections over time into stable track IDs with a simple motion
   model (constant-velocity smoother) and a configurable miss/hit threshold.
3. **Alert** when a track meets configurable rules (e.g. confidence over
   threshold for N consecutive frames, dwell time, distance from a defined
   keep-out polygon). Alerts are scored by severity (info/low/med/high).
4. **Incidents**: persist alerts, the originating track, a short clip
   reference, and operator notes. Incidents have a status lifecycle
   (`new -> acknowledged -> reviewed -> closed`).
5. **Replay**: reconstruct a track on the timeline view from stored frames or
   metadata for after-the-fact review.
6. **Operator UI**: live monitor with detection overlays, alert list with
   filtering, incident replay view, exportable incident summary (JSON + a
   human-readable Markdown summary), and an audit log view.
7. **Audit log**: every operator action (acknowledge, change status, add note,
   export, login) is recorded with user id, timestamp, and a hash chain so
   tampering is detectable.
8. **Health endpoints** on every service (`/healthz`, `/readyz`).

## Out of scope — explicitly forbidden

These are hard product boundaries. They are not "nice-to-haves to defer";
they are explicitly **not** part of Maple Shield, ever.

- **No interception, neutralization, jamming, spoofing, take-over, or
  damage** of any drone, RF link, GPS link, or operator.
- **No targeting, weapons, or weapon-control workflows** of any kind. No
  integration interfaces, no pluggable backends, no "stub" hooks for any of
  the above.
- **No autonomous engagement.** The platform never initiates an action against
  a detected object. All operator actions are human-initiated and logged.
- **No instructions, code, configuration, or documentation** for defeating,
  damaging, disabling, hijacking, or interfering with drones.
- **No facial recognition or biometric identification** of operators or
  bystanders. Detection is restricted to the drone object class.
- **No covert surveillance of private property** outside the deploying
  organisation's lawful monitoring zone. Camera placement and retention
  policy are the operator's responsibility; the platform provides defaults
  that favour minimisation (see `docs/COMPLIANCE.md`).

## Non-goals (out of MVP, may revisit)

- Multi-sensor fusion (RF, radar, acoustic). MVP is electro-optical only.
- Federation across sites.
- Mobile clients.
- ML model training pipeline. MVP uses a stub or pre-trained off-the-shelf
  detector behind a stable interface; training is not part of the product.

## Success criteria for the MVP

- Operator can start the stack with one command and see a sample video
  flow end-to-end (ingest -> detection -> track -> alert -> incident -> review).
- All five components have an `/healthz` endpoint that returns 200 when
  ready.
- Audit log records every state-changing operator action and the chain
  verifies.
- All unit tests pass (`pytest`) and the UI builds (`npm run build`) clean.
- `docs/COMPLIANCE.md` enumerates the boundary checks; `tests/` includes a
  test that fails if any forbidden interface is added.

## Assumptions

- Operators are trained and authorised under their jurisdiction's airspace
  monitoring rules.
- Camera coverage and retention is configured by the deploying organisation
  in accordance with local privacy law.
- The platform is deployed inside a controlled network; authentication and
  TLS are provided by the deployment environment in production.
