# BOREALIS Testing Strategy

## Unit Tests
- NMS post-processing (test with known boxes)
- IoU calculation (edge cases: zero overlap, full overlap)
- Risk scoring (boundary conditions)

## Integration Tests
- Full pipeline (dummy video → detections.jsonl)
- Replay integrity (raw.mp4 + JSONL → same output)

## Performance Tests
- FPS benchmark (automated, compare against baseline)
- Memory leak detection (long-running test)

## Regression Tests
- Model swap (does new ONNX maintain performance?)
