"""
maple_shield_mvp.py — Maple Shield Detection MVP

Full pipeline:
  EO video → YOLOv8 ONNX inference (sparse-optimised where GPU available)
  → IoU multi-target tracker → Threat scorer → Overlay dashboard → JSONL log

Shape gate automatically selects dense/sparse kernel based on M_eff.
On CPU (no CUDA) runs dense ONNX — identical results, portable everywhere.

Usage:
    python maple_shield_mvp.py                        # webcam
    python maple_shield_mvp.py --source video.mp4     # video file
    python maple_shield_mvp.py --model models/yolov8n_sparse24.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from shape_gate import gate, KernelMode, report as gate_report, GateConfig
from threat_scorer import (
    ScorerConfig, ThreatLevel, score_detection, ScoredDetection
)
from maple_shield_mqtt import AlertPublisher
from maple_shield_cot import CotPublisher, CotConfig

# ---------------------------------------------------------------------------
# COCO labels
# ---------------------------------------------------------------------------

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONF_THRES       = 0.35
IOU_THRES        = 0.45
MAX_DET          = 50
IMGSZ            = 640
TRACK_IOU_THRES  = 0.30
TRACK_MAX_AGE    = 30
TRACK_MIN_HITS   = 2
TRACK_SMOOTH_A   = 0.70
VIDEO_FPS        = 30.0
LOG_EVERY_N      = 1

# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, size: int = 640) -> np.ndarray:
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)


def xywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        iou = (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0) * \
              (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0) / \
              (areas[i] + areas[order[1:]] - \
               (np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])).clip(0) * \
               (np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]])).clip(0) + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


def postprocess(out: np.ndarray, conf_thresh=0.35, iou_thresh=0.45, max_det=50) -> np.ndarray:
    pred = out[0].transpose(1, 0).astype(np.float32)
    cls_id = np.argmax(pred[:, 4:], axis=1)
    conf = pred[:, 4:][np.arange(pred.shape[0]), cls_id]
    mask = conf >= conf_thresh
    boxes, conf, cls_id = pred[mask, :4], conf[mask], cls_id[mask]
    if conf.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    xyxy = xywh_to_xyxy(boxes)
    keep = nms(xyxy, conf, iou_thresh)[:max_det]
    return np.stack([xyxy[keep, 0], xyxy[keep, 1], xyxy[keep, 2], xyxy[keep, 3],
                     conf[keep], cls_id[keep].astype(np.float32)], axis=1)


def scale_boxes(det: np.ndarray, imgsz: int, fw: int, fh: int) -> np.ndarray:
    if det.shape[0] == 0: return det
    d = det.copy()
    d[:, 0::2] = np.clip(d[:, 0::2] * fw / imgsz, 0, fw - 1)
    d[:, 1::2] = np.clip(d[:, 1::2] * fh / imgsz, 0, fh - 1)
    return d


# ---------------------------------------------------------------------------
# IoU Tracker
# ---------------------------------------------------------------------------

class Track:
    def __init__(self, tid, box, cls, conf, label):
        self.id, self.box, self.cls = tid, np.array(box, dtype=np.float32), int(cls)
        self.conf, self.label = float(conf), label
        self.hits, self.age, self.confirmed = 1, 0, False
        self.prev_center: tuple[float, float] | None = None

    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    def update(self, box, conf, alpha=0.7):
        self.prev_center = self.center()
        self.box = alpha * self.box + (1 - alpha) * np.array(box, dtype=np.float32)
        self.conf = float(conf); self.hits += 1; self.age = 0
        if self.hits >= TRACK_MIN_HITS: self.confirmed = True

    def mark_missed(self): self.age += 1


class IoUTracker:
    def __init__(self):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def _iou(self, a, b) -> float:
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
        ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
        return inter / (aa + ab - inter + 1e-6)

    def step(self, det_list: list[dict]) -> list[dict]:
        for t in self.tracks.values(): t.mark_missed()
        assigned = set()
        for d in sorted(det_list, key=lambda x: x["conf"], reverse=True):
            best_tid, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in assigned or t.cls != d["cls"]: continue
                iou = self._iou(t.box.tolist(), d["box"])
                if iou > best_iou: best_iou = iou; best_tid = tid
            if best_tid and best_iou >= TRACK_IOU_THRES:
                t = self.tracks[best_tid]
                t.update(d["box"], d["conf"], TRACK_SMOOTH_A)
                assigned.add(best_tid)
                d["track_id"] = int(t.id); d["track_confirmed"] = bool(t.confirmed)
                d["prev_center"] = t.prev_center
            else:
                tid = self.next_id; self.next_id += 1
                t = Track(tid, d["box"], d["cls"], d["conf"], d["label"])
                self.tracks[tid] = t; assigned.add(tid)
                d["track_id"] = int(tid); d["track_confirmed"] = False
                d["prev_center"] = None
        for tid in [tid for tid, t in self.tracks.items() if t.age > TRACK_MAX_AGE]:
            del self.tracks[tid]
        return det_list


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def draw_zone(frame: np.ndarray, config: ScorerConfig) -> np.ndarray:
    cx, cy = config.frame_w // 2, config.frame_h // 2
    r = int(config.zone_radius_frac * min(config.frame_w, config.frame_h))
    cv2.circle(frame, (cx, cy), r, (50, 50, 200), 1)
    cv2.putText(frame, "ZONE", (cx - r + 5, cy - r + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 200), 1)
    return frame


def draw_detection(frame: np.ndarray, sd: ScoredDetection) -> np.ndarray:
    x1, y1, x2, y2 = sd.box
    color = sd.threat_level.color_bgr()
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"ID{sd.track_id} {sd.label} {sd.conf:.2f} [{sd.threat_level.label()}]"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    ty = max(0, y1 - 6)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
    cv2.putText(frame, tag, (x1 + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def draw_hud(frame: np.ndarray, fps: float, infer_ms: float,
             n_det: int, kernel_mode: KernelMode, max_threat: ThreatLevel) -> np.ndarray:
    mode_str = f"SPARSE 2:4" if kernel_mode == KernelMode.SPARSE else "DENSE (CPU)"
    hud = (f"FPS {fps:.1f}  |  infer {infer_ms:.1f}ms  |  det {n_det}"
           f"  |  {mode_str}  |  MAX: {max_threat.label()}")
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)
    # Threat indicator bar (top-right)
    bar_colors = {
        ThreatLevel.CLEAR:    (0, 180, 0),
        ThreatLevel.LOW:      (0, 200, 200),
        ThreatLevel.MEDIUM:   (0, 165, 255),
        ThreatLevel.HIGH:     (0, 0, 255),
        ThreatLevel.CRITICAL: (0, 0, 180),
    }
    bx = frame.shape[1] - 140
    for i, lvl in enumerate(ThreatLevel):
        c = bar_colors[lvl] if lvl <= max_threat else (60, 60, 60)
        cv2.rectangle(frame, (bx + i * 26, 4), (bx + i * 26 + 22, 32), c, -1)
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(source, model_path: str, runs_dir: Path):
    # Gate report
    gr = gate_report()
    print("=== Maple Shield MVP ===")
    print(f"  GPU/Sparse capable : {gr['sparse_capable']}")
    print(f"  SM version         : {gr['sm_version']}")
    print(f"  Expected gain      : {gr['expected_speedup_ffn_down']}")
    print()

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_dir = runs_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if gr["sparse_capable"] else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_file), providers=providers)
    inp_name = sess.get_inputs()[0].name
    print(f"  Model    : {model_file.name}")
    print(f"  Provider : {sess.get_providers()[0]}")

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")
    fh, fw = frame0.shape[:2]

    writer = cv2.VideoWriter(
        str(run_dir / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (fw, fh)
    )

    scorer_cfg = ScorerConfig(frame_w=fw, frame_h=fh)
    gate_cfg   = GateConfig()
    tracker    = IoUTracker()

    # MQTT alert publisher (graceful no-op if broker unavailable)
    mqtt_pub = AlertPublisher(
        session_id=run_dir.name,
        frame_w=fw, frame_h=fh,
    )
    mqtt_pub.connect()

    # CoT UDP publisher (ATAK / WinTAK / TAK Server)
    cot_pub = CotPublisher(CotConfig(frame_w=fw, frame_h=fh))
    cot_pub.start()

    log_path = run_dir / "detections.jsonl"
    log_f = open(log_path, "w", encoding="utf-8")

    (run_dir / "meta.json").write_text(json.dumps({
        "model": str(model_file),
        "provider": sess.get_providers()[0],
        "sparse_capable": gr["sparse_capable"],
        "frame_w": fw, "frame_h": fh,
        "started_ts": time.time(),
    }, indent=2))

    frame_id = 0
    t0 = time.time()
    kernel_mode = KernelMode.DENSE

    print(f"\nStreaming → press Q to quit | run: {run_dir}\n")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_id += 1
        ts = time.time()

        inp = preprocess(frame, IMGSZ)
        # Shape gate decision (M = batch * spatial tokens for transformer-style models)
        M_eff = IMGSZ  # proxy for spatial resolution
        kernel_mode = gate(M_eff, gate_cfg)

        t_inf = time.time()
        raw = sess.run(None, {inp_name: inp})[0]
        infer_ms = (time.time() - t_inf) * 1000.0

        det = postprocess(raw, CONF_THRES, IOU_THRES, MAX_DET)
        det = scale_boxes(det, IMGSZ, fw, fh)

        det_list = [{
            "cls": int(c), "label": COCO80[int(c)] if 0 <= int(c) < len(COCO80) else f"cls{int(c)}",
            "conf": float(s), "box": [int(x1), int(y1), int(x2), int(y2)]
        } for x1, y1, x2, y2, s, c in det]

        det_list = tracker.step(det_list)

        scored: list[ScoredDetection] = []
        for d in det_list:
            sd = score_detection(
                track_id=d["track_id"],
                label=d["label"],
                conf=d["conf"],
                box=d["box"],
                track_confirmed=d["track_confirmed"],
                prev_center=d.get("prev_center"),
                config=scorer_cfg,
            )
            scored.append(sd)

        max_threat = max((s.threat_level for s in scored), default=ThreatLevel.CLEAR)
        fps = frame_id / max(1e-6, ts - t0)

        # Render
        draw_zone(frame, scorer_cfg)
        for sd in scored:
            if sd.object_type != "ignore":
                draw_detection(frame, sd)
        draw_hud(frame, fps, infer_ms, len(scored), kernel_mode, max_threat)

        writer.write(frame)
        cv2.imshow("MAPLE SHIELD — Edge AI Drone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # MQTT: publish threat escalation alerts to C2
        mqtt_pub.on_frame(frame_id, scored, max_threat.label(), fps, kernel_mode.name)

        # CoT: broadcast NATO-compatible CoT XML to ATAK / TAK Server
        cot_pub.on_frame(frame_id, scored, run_dir.name, fps)

        if frame_id % LOG_EVERY_N == 0:
            log_f.write(json.dumps({
                "ts": ts, "frame": frame_id, "infer_ms": infer_ms, "fps": fps,
                "kernel_mode": kernel_mode.name,
                "max_threat": max_threat.label(),
                "detections": [{
                    "track_id": s.track_id, "label": s.label, "object_type": s.object_type,
                    "conf": s.conf, "box": s.box, "confirmed": s.track_confirmed,
                    "threat": s.threat_level.label(), "score": s.threat_score,
                    "in_zone": s.in_zone, "velocity_px": s.velocity_px,
                } for s in scored],
            }) + "\n")

    log_f.close(); cap.release(); writer.release(); cv2.destroyAllWindows()
    mqtt_pub.disconnect()
    cot_pub.stop()
    print(f"\nRun saved → {run_dir}")
    print(f"  MQTT alerts published : {mqtt_pub.alerts_published}")
    print(f"  CoT events sent       : {cot_pub.events_sent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maple Shield Detection MVP")
    parser.add_argument("--source", default=0, help="Webcam index or video path")
    parser.add_argument("--model", default=r"models\maple_shield_drone_v1_dense.onnx", help="ONNX model path")
    parser.add_argument("--runs-dir", default="runs", help="Output directory")
    args = parser.parse_args()

    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    main(src, args.model, Path(args.runs_dir))
