# Maple Shield - NATO IC26 Repo-Aligned Response

Source document reviewed: `C:\Users\15879\Downloads\Maple Shield NATO IC26 Final.pdf`

Alignment basis: PR #13, commit `228b8d5`, plus the repo evidence in
`docs/DOC_REPO_ALIGNMENT.md`, `docs/DEMO_VALIDATION.md`, `scripts/benchmark.py`,
`run_maple_shield_demo.py`, `maple_shield_cairn_session.py`, and the platform
test suites.

## 1. Executive Summary

Maple Shield is a software-first, passive airspace monitoring and drone
early-warning platform built around the CAIRN detection and risk engine. The
current repo supports passive observation, drone-only detection contracts,
operator alerting, audit-ready records, and replayable incident workflows.

The current demonstrable capability is strongest in the sensing, scoring,
operator console, and evidence layer:

- Passive, vision-first product boundary enforced by compliance tests.
- CAIRN risk scoring over deterministic drone detections.
- `cairn.frame.v1` JSONL records for audit and replay.
- Operator UI with Live, Alerts, Replay, and Audit views.
- Repeatable benchmark and one-command demo that measure the CAIRN scoring
  stage on host CPU.

Maple Shield should be positioned for IC26 as a passive sensing and evidence
layer that complements existing counter-UAS architectures. It should not yet
claim field-proven Jetson performance, thermal fusion validation, STANAG 4609
conformance, Arctic validation, mesh coordination, or sub-50 ms full
detection-to-alert latency until those measurements exist.

## 2. Strategic Context

Small UAS proliferation makes low-cost airspace awareness strategically
important. Maple Shield addresses the detection and operator-awareness side of
that problem by emphasizing:

- Passive sensing where radar or RF-dependent systems are unavailable,
  undesirable, or already saturated.
- Software-defined model and scoring updates.
- Edge-first operation for constrained environments.
- Evidence generation through structured frame records, incident timelines,
  and hash-chained audit logs.

Repo-backed wording:

> Maple Shield is being developed as a passive visual sensing and operator
> awareness layer for layered C-UAS environments. Its near-term value is
> detection, classification support, tracking metadata, audit records, and
> operator alerting.

Avoid claiming:

- NATO-wide deployment economics.
- Specific deployment density improvements.
- Field-proven cold-weather operation.
- Field-proven cost-per-node numbers.
- Downstream mitigation or defeat capability.

## 3. Problem Statement

Current C-UAS architectures can be costly, difficult to deploy densely, and
dependent on sensor modalities that are not always desirable in restricted or
emissions-conscious environments. Maple Shield targets a narrower and more
defensible problem:

> Can a low-cost, passive, edge-first software layer provide useful drone
> detection evidence and operator alerts without claiming to replace radar,
> RF detection, or full-spectrum C-UAS systems?

Repo evidence supports:

- Passive product boundary.
- Drone-only class contracts in the platform.
- Structured event and audit records.
- Replayable operator workflows.
- CAIRN scoring and benchmark outputs.

Repo evidence does not yet support:

- 15 km perimeter coverage.
- Live outdoor multi-node trials.
- -40 C field validation.
- Under-25 W power validation.
- Real-camera end-to-end latency.
- False-positive rates on labelled field data.

## 4. Solution Description

### 4.1 Current Demonstrable System

Current repo evidence supports this description:

> Maple Shield combines a passive drone-detection software architecture, a
> CAIRN risk-scoring engine, an edge-agent interface, a fusion/alerting layer,
> and an operator console. The current proof layer uses deterministic synthetic
> drone detections to validate CAIRN scoring, frame-record output, demo gating,
> and operator audit flows.

Current components:

- `cairn_engine/`: CAIRN schemas and risk scoring.
- `platform/edge-agent/`: detector interfaces and CAIRN adapter.
- `platform/fusion/`: track and alert scoring.
- `platform/command-api/`: incident, replay, and audit APIs.
- `platform/operator-ui/`: Live, Alerts, Replay, and Audit UI.
- `scripts/benchmark.py`: repeatable CAIRN scoring-stage benchmark.
- `run_maple_shield_demo.py`: one-command demo with PASS/FAIL gates.
- `maple_shield_cairn_session.py`: opt-in helper for JSONL frame records.

### 4.2 Performance Claims

Use only these claims externally unless newer measured evidence lands:

| Capability | Repo-aligned wording |
|---|---|
| FPS | CAIRN scoring stage runs above the demo PASS threshold on host CPU. Cite the relevant `summary.json`. |
| Latency | CAIRN scoring-stage p95 latency is measured by `summary.json::cairn_scoring_ms.p95`. |
| Full pipeline latency | Roadmap. Not measured by PR #13. |
| Jetson Orin Nano | Roadmap/in progress. Config and profiling hooks exist, but no measured Jetson run is in the repo. |
| Thermal/IR | Roadmap. Scaffolds exist, but no validation evidence. |
| False-positive rate | Roadmap. No labelled field evaluation in repo. |
| Power draw | Roadmap. No measured power data in repo. |
| STANAG 4609 | Roadmap. Scaffold exists, but no conformance validation. |

### 4.3 Safe Target Specification Table

| Parameter | Repo-aligned status |
|---|---|
| Passive detection and alerting | Proven for software boundary and platform contracts. |
| CAIRN frame records | Proven by tests and benchmark JSONL. |
| Operator console and replay | Proven by platform tests, build, and browser QA summary. |
| Host CPU scoring benchmark | Demo-supported by `scripts/benchmark.py`. |
| Real YOLO inference through CAIRN | In progress. Adapter exists; live end-to-end wiring remains next upgrade. |
| Jetson 30+ FPS | Roadmap until measured on Jetson hardware. |
| Less than 50 ms full detection-to-alert | Roadmap until measured end-to-end. |
| Less than 2 percent false positives | Roadmap until labelled evaluation exists. |
| Arctic -40 C validation | Roadmap until field/environmental testing exists. |
| Multi-node mesh | Roadmap until exercised by tests or demo. |

## 5. Threat Coverage

Repo-aligned phrasing:

> Maple Shield is intended to support detection workflows for micro, Class I,
> and Class II UAS categories. Current repo proof uses deterministic drone
> detections and CAIRN scoring. Field coverage by UAS class, detection range,
> degraded-weather performance, and swarm-scale performance remain validation
> objectives for IC26 trials.

Avoid presenting class-specific ranges, night capability, or multi-node swarm
handling as proven unless backed by new field data.

## 6. System Architecture

Current repo-backed architecture:

| Layer | Evidence |
|---|---|
| Detection interface | `platform/edge-agent/edge_agent/detector.py` and CAIRN adapter. |
| CAIRN risk scoring | `cairn_engine/`, `tests/test_cairn_session.py`. |
| Benchmark/demo | `scripts/benchmark.py`, `run_maple_shield_demo.py`. |
| Fusion/alerts | `platform/fusion/`, platform e2e tests. |
| Command API | `platform/command-api/`, audit-chain tests. |
| Operator UI | `platform/operator-ui/`, Vite build, browser QA summary. |
| Audit/evidence | `cairn.frame.v1` JSONL, incident replay, hash-chained audit log. |

Repo-aligned C2 wording:

> Maple Shield emits structured JSON records suitable for integration work and
> future C2 adapters. CoT and STANAG-compatible paths are development targets;
> STANAG conformance is not yet validated.

## 7. NATO IC26 Theme Alignment

### 7.1 Layered Counter-UAS Initiative

Repo-aligned claim:

> Maple Shield contributes a passive visual sensing and evidence layer to
> layered C-UAS architectures. It is designed to complement, not replace,
> existing radar, RF, and authorized response systems.

### 7.2 High North / Arctic Operations

Repo-aligned claim:

> Maple Shield is architecturally relevant to remote and High North operations
> because it is edge-first and does not depend on cloud services for the current
> proof path. Arctic temperature, power, and remote-autonomy claims remain
> validation objectives.

### 7.3 AI Next Generation C4ISR

Repo-aligned claim:

> Maple Shield produces structured machine-readable detection and risk records,
> including confidence, scoring metadata, reasons, and operator-facing alert
> context.

### 7.4 Next Generation Targeting

Safer repo-aligned replacement:

> Maple Shield can provide operator awareness and cueing metadata to authorized
> downstream systems. The repo does not implement autonomous response logic or
> mitigation control.

### 7.5 Electronic Warfare

Safer repo-aligned replacement:

> Maple Shield's sensing concept does not rely on GPS or RF detection. It can
> therefore remain useful in RF-denied contexts for visual observation, subject
> to optical line-of-sight, weather, camera, power, and networking limits.

Avoid absolute claims such as "completely immune to electronic attack."

## 8. Competitive Positioning

Repo-aligned positioning:

> Maple Shield is an additive software intelligence layer. Its credible near-term
> wedge is lower-cost passive observation, structured evidence, and operator
> workflow integration. It should be compared as a sensing and evidence layer,
> not as a full-spectrum C-UAS system.

Avoid unsupported comparison rows:

- Full Arctic operation.
- Confirmed under-25 W node power.
- Confirmed under-$1K deployed hardware.
- Confirmed 20+ targets per node.
- Confirmed STANAG data feed.
- Confirmed day/night thermal operation.

## 9. Risk Assessment and Mitigation

Repo-aligned risk table:

| Risk | Current mitigation | Evidence status |
|---|---|---|
| False positives | Drone-only platform contracts, CAIRN scoring, replay/audit workflow. | Software-supported, field rate unmeasured. |
| Adverse weather | Thermal/IR fusion listed as roadmap. | Not validated. |
| Camera obstruction | Operator health/status concepts exist in platform direction. | Needs live deployment evidence. |
| Overclaiming | `docs/DOC_REPO_ALIGNMENT.md` and `docs/DEMO_VALIDATION.md` constrain public claims. | Proven by PR #13 docs/tests. |
| Integration drift | Structured JSONL and tests pin schema shape. | Proven for current schema. |

## 10. Technology Readiness

Repo-aligned TRL wording:

> Maple Shield currently has a software proof layer with validated component
> behavior in repo: CAIRN scoring, structured frame records, demo output,
> operator UI, alert/replay flows, and compliance boundaries. Outdoor prototype,
> Jetson, thermal, mesh, Arctic, and STANAG validation remain the next
> readiness steps.

Suggested table:

| Milestone | Repo-aligned status |
|---|---|
| Software concept and schemas | Complete in repo. |
| CAIRN scoring and JSONL frame records | Complete in repo. |
| Operator UI and replay | Complete in repo. |
| Repeatable host-CPU scoring demo | Complete in PR #13. |
| Real YOLO through CAIRN | In progress. |
| Outdoor live-drone prototype | Planned. |
| Jetson measurement | Planned. |
| Multi-node mesh | Planned. |
| NATO operational demonstration | Target through IC26. |

## 11. Requested IC26 Engagement

Repo-aligned request:

Maple Silicon Inc. seeks IC26 support for validation, not for unsupported
claims. The highest-value activities are:

- Controlled live-drone testing against representative micro, Class I, and
  Class II UAS targets.
- Measurement of full detection-to-alert latency on real video.
- Jetson Orin Nano performance and power profiling.
- False-positive evaluation against labelled bird, debris, weather, and drone
  datasets.
- Integration feedback for structured JSONL, CoT, and future STANAG adapters.
- Environmental testing for High North and cold-weather operating assumptions.

## 12. Conclusion

Repo-aligned conclusion:

Maple Shield's strongest current case is not that every field-performance
number is already proven. Its strongest case is that the software foundation is
now measurable, testable, and honest: CAIRN scoring, frame records, operator
workflow, auditability, and claim discipline are in the repo, while hardware,
field, and environmental claims are explicitly tracked as validation objectives.

That makes Maple Shield a credible candidate for IC26 experimentation: the
program can help convert repo-backed software proof into measured operational
evidence.

