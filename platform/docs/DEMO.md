# Maple Shield Demo

This demo starts the local Maple Shield stack on one laptop:

- command-api on `http://localhost:8080`
- fusion-engine on `http://localhost:8090`
- edge-agent with either the mock detector or a YOLO ONNX provider
- operator-ui on `http://localhost:5173`

The default path uses the deterministic mock feed. That means the demo works
without a camera, video file, or model weights.

## Quick Start

From `platform/` on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\demo_live.ps1
```

From `platform/` on macOS/Linux:

```bash
bash scripts/demo_live.sh
```

Or use:

```bash
make demo
```

To preview the commands through Make without starting services:

```bash
make demo DEMO_ARGS=--dry-run
```

## Smoke Test

Use the bounded smoke runner when you want proof that the local service path
works without keeping the UI open:

```bash
python scripts/smoke_demo.py
```

Or, where `make` is available:

```bash
make smoke-demo
```

The smoke runner starts command-api and fusion-engine on random local ports,
runs the mock edge-agent for a short burst, verifies alerts, incidents, and the
audit chain, then shuts the services down.

Logs are written to `platform/.demo-logs/`. Press `Ctrl+C` in the launcher
terminal to stop all demo processes.

## Browser QA Evidence

The latest browser-visible route check is documented in
`platform/docs/DEMO_QA.md`. It records the local stack URLs, seeded mock feed,
API evidence, and the operator UI routes verified with headless Chrome.

Use `http://127.0.0.1:5173` for this QA pass when another local Vite app is
already bound to the IPv6 `localhost` route.

## Dry Run

Use dry run mode when you only want to inspect the commands:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\demo_live.ps1 -DryRun
```

```bash
bash scripts/demo_live.sh --dry-run
```

## YOLO ONNX Mode

Use this when you have a local ONNX model whose labels include `drone`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\demo_live.ps1 `
  -Source C:\path\to\sample.mp4 `
  -Detector cairn-yolo `
  -YoloModel C:\path\to\drone-detector.onnx `
  -YoloClasses drone
```

```bash
bash scripts/demo_live.sh \
  --source /path/to/sample.mp4 \
  --detector cairn-yolo \
  --yolo-model /path/to/drone-detector.onnx \
  --yolo-classes drone
```

If the source file or model is missing, the launcher falls back to the mock
feed so the operator UI still opens and the full event path can be checked.

## What To Look For

Open `http://localhost:5173` and check:

- Live view shows track updates as the mock feed advances.
- Alerts appear as the confidence ladder rises.
- Replay and audit views remain available for operator review.
- The CAIRN status panel stays readable even when the demo uses mock data.

## Troubleshooting

- If a service fails to start, check `platform/.demo-logs/*.err.log`.
- If the UI opens but shows no tracks, confirm fusion-engine is healthy at
  `http://localhost:8090/healthz`.
- If the YOLO path produces no detections, confirm the label map includes
  `drone` and pass it with `--yolo-classes drone` or `-YoloClasses drone`.
- If ports are already in use, stop the old local demo processes and rerun.
