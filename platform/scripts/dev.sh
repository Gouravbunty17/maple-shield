#!/usr/bin/env bash
# Spin up the dev stack. Each service in the background; trap to clean up.
set -e
cd "$(dirname "$0")/.."
PIDS=()
cleanup() { echo; echo "Stopping..."; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup INT TERM EXIT

echo "[command-api] :8080"
( cd command-api && uvicorn app.main:app --port 8080 --reload ) &
PIDS+=($!)

sleep 1

echo "[fusion-engine] :8090"
( cd fusion-engine && uvicorn fusion.main:app --port 8090 --reload ) &
PIDS+=($!)

sleep 1

echo "[edge-agent] mock feed"
( cd edge-agent && python -m edge_agent.main --source mock --fusion http://localhost:8090 ) &
PIDS+=($!)

echo "[operator-ui] :5173"
( cd operator-ui && npm run dev ) &
PIDS+=($!)

wait
