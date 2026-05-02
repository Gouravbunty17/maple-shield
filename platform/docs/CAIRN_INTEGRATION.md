# CAIRN Integration

Maple Shield platform Phase 2 adds a narrow adapter between the existing
`cairn_engine` package and the platform edge-agent contract.

## Boundary

The adapter is passive. It converts CAIRN frame results into platform detection
metadata and drops every non-drone class before anything reaches fusion.

## Data Flow

```text
CairnDetection provider
  -> CairnEngine.process_frame()
  -> CairnFrameRecord.risks
  -> CairnSourceDetector
  -> platform Detection rows
  -> fusion-engine
  -> command-api
  -> operator-ui
```

## Translation Rules

- `CairnRiskResult.box` becomes `Detection.bbox`.
- `CairnRiskResult.confidence` becomes `Detection.confidence`.
- `CairnRiskResult.track_id` becomes `Detection.track_id` with a `cairn-` prefix.
- CAIRN labels are classified through `CairnRiskEngine.classify_object`.
- Only `object_type == "drone"` is emitted.
- CAIRN risk fields are stored as local display hints in `Detection.raw`.
- Platform severity is still owned by `fusion-engine`.

## Health Endpoint

`GET /cairn/health` exposes:

- CAIRN engine name and version
- adapter compatibility check
- frames processed
- runtime
- risk configuration

No operator notes, imagery, or incident content are included.

## Testing

The integration is covered by:

- `edge-agent/tests/test_cairn_adapter.py`
- `command-api/tests/test_cairn_health.py`
- `tests/test_e2e_cairn.py`
- the existing compliance test suite
