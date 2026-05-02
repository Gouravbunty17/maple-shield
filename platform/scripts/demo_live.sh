#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/.demo-logs"
SOURCE="mock"
DETECTOR="mock"
YOLO_MODEL="${MAPLE_SHIELD_YOLO_MODEL:-}"
YOLO_CLASSES="${MAPLE_SHIELD_YOLO_CLASSES:-}"
FRAMES="180"
FPS="10"
NO_BROWSER="false"
DRY_RUN="false"
PIDS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --detector) DETECTOR="$2"; shift 2 ;;
    --yolo-model) YOLO_MODEL="$2"; shift 2 ;;
    --yolo-classes) YOLO_CLASSES="$2"; shift 2 ;;
    --frames) FRAMES="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --no-browser) NO_BROWSER="true"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

cleanup() {
  echo "[demo] stopping..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

if [[ "$SOURCE" != "mock" && ! -e "$SOURCE" ]]; then
  echo "[demo] source '$SOURCE' was not found; falling back to mock."
  SOURCE="mock"
fi

if [[ "$DETECTOR" == "cairn-yolo" && -z "$YOLO_MODEL" ]]; then
  echo "[demo] no YOLO model was provided; falling back to mock."
  DETECTOR="mock"
fi

EDGE_CMD=(python -m edge_agent.main --source "$SOURCE" --detector "$DETECTOR" --n-frames "$FRAMES" --fps "$FPS" --fusion http://localhost:8090)
if [[ "$DETECTOR" == "cairn-yolo" ]]; then
  EDGE_CMD+=(--yolo-model "$YOLO_MODEL")
  if [[ -n "$YOLO_CLASSES" ]]; then
    EDGE_CMD+=(--yolo-classes "$YOLO_CLASSES")
  fi
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[demo] root: $ROOT"
  echo "[demo] command-api: python -m uvicorn app.main:app --port 8080"
  echo "[demo] fusion-engine: python -m uvicorn fusion.main:app --port 8090"
  echo "[demo] edge-agent: ${EDGE_CMD[*]}"
  echo "[demo] operator-ui: npm run dev -- --host 127.0.0.1"
  trap - INT TERM EXIT
  exit 0
fi

wait_health() {
  local name="$1"
  local url="$2"
  for _ in $(seq 1 30); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[$name] healthy"
      return 0
    fi
    sleep 0.5
  done
  echo "[$name] did not become healthy at $url" >&2
  exit 1
}

mkdir -p "$LOG_DIR"
export MAPLE_SHIELD_LAWFUL_USE_ACK=true

(cd "$ROOT/command-api" && python -m uvicorn app.main:app --port 8080 >"$LOG_DIR/command-api.out.log" 2>"$LOG_DIR/command-api.err.log") &
PIDS+=("$!")
wait_health "command-api" "http://localhost:8080/healthz"

(cd "$ROOT" && python -m uvicorn fusion.main:app --port 8090 >"$LOG_DIR/fusion-engine.out.log" 2>"$LOG_DIR/fusion-engine.err.log") &
PIDS+=("$!")
wait_health "fusion-engine" "http://localhost:8090/healthz"

(cd "$ROOT/edge-agent" && "${EDGE_CMD[@]}" >"$LOG_DIR/edge-agent.out.log" 2>"$LOG_DIR/edge-agent.err.log") &
PIDS+=("$!")

(cd "$ROOT/operator-ui" && npm run dev -- --host 127.0.0.1 >"$LOG_DIR/operator-ui.out.log" 2>"$LOG_DIR/operator-ui.err.log") &
PIDS+=("$!")

echo "[demo] logs: $LOG_DIR"
echo "[demo] UI: http://localhost:5173"
if [[ "$NO_BROWSER" != "true" ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open http://localhost:5173 >/dev/null 2>&1 || true
  elif command -v open >/dev/null 2>&1; then
    open http://localhost:5173 >/dev/null 2>&1 || true
  fi
fi
echo "[demo] Press Ctrl+C to stop."
wait
