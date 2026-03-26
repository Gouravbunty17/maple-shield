import json
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# -----------------------
# Config
# -----------------------
MODEL_PATH = r"models\yolov8n.onnx"
RUNS_DIR = Path("runs")
IMGSZ = 640

CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_DET = 50

SAVE_OVERLAY_VIDEO = True
VIDEO_FPS = 30.0

LOG_EVERY_N_FRAMES = 1
SAVE_FRAME_HASH = False

# Tracking params (IoU tracker)
TRACK_IOU_THRES = 0.30
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 2
TRACK_SMOOTH_ALPHA = 0.70

# -----------------------
# Utils
# -----------------------
def now_s():
    return time.time()

def preprocess_bgr(frame, size=640):
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def xywh_to_xyxy(xywh):
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)

def nms_xyxy(boxes, scores, iou_thresh=0.45):
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def postprocess_yolov8(out, conf_thresh=0.35, iou_thresh=0.45, max_det=50):
    pred = out[0].transpose(1, 0).astype(np.float32)
    boxes_xywh = pred[:, 0:4]
    cls_scores = pred[:, 4:]
    cls_id = np.argmax(cls_scores, axis=1)
    conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

    mask = conf >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if conf.size == 0:
        return np.zeros((0,6), dtype=np.float32)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)
    keep = nms_xyxy(boxes_xyxy, conf, iou_thresh=iou_thresh)[:max_det]

    det = np.stack([
        boxes_xyxy[keep, 0],
        boxes_xyxy[keep, 1],
        boxes_xyxy[keep, 2],
        boxes_xyxy[keep, 3],
        conf[keep],
        cls_id[keep].astype(np.float32)
    ], axis=1)
    return det

def scale_boxes(det, imgsz, frame_w, frame_h):
    if det.shape[0] == 0:
        return det
    sx = frame_w / float(imgsz)
    sy = frame_h / float(imgsz)
    d = det.copy()
    d[:, 0] = np.clip(d[:, 0] * sx, 0, frame_w - 1)
    d[:, 1] = np.clip(d[:, 1] * sy, 0, frame_h - 1)
    d[:, 2] = np.clip(d[:, 2] * sx, 0, frame_w - 1)
    d[:, 3] = np.clip(d[:, 3] * sy, 0, frame_h - 1)
    return d

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)

# -----------------------
# Simple IoU Tracker
# -----------------------
class Track:
    def __init__(self, track_id, box, cls, conf, label):
        self.id = track_id
        self.box = np.array(box, dtype=np.float32)
        self.cls = int(cls)
        self.conf = float(conf)
        self.label = label
        self.hits = 1
        self.age = 0
        self.confirmed = False

    def update(self, box, conf, alpha=0.7):
        box = np.array(box, dtype=np.float32)
        self.box = alpha * self.box + (1 - alpha) * box
        self.conf = float(conf)
        self.hits += 1
        self.age = 0
        if self.hits >= TRACK_MIN_HITS:
            self.confirmed = True

    def mark_missed(self):
        self.age += 1

class IoUTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def step(self, det_list):
        for t in self.tracks.values():
            t.mark_missed()

        dets = sorted(det_list, key=lambda d: d["conf"], reverse=True)

        assigned_tracks = set()

        for d in dets:
            best_tid = None
            best_iou = 0.0

            for tid, t in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                if t.cls != d["cls"]:
                    continue

                i = iou_xyxy(t.box.tolist(), d["box"])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_tid is not None and best_iou >= TRACK_IOU_THRES:
                t = self.tracks[best_tid]
                t.update(d["box"], d["conf"], alpha=TRACK_SMOOTH_ALPHA)
                assigned_tracks.add(best_tid)
                d["track_id"] = int(t.id)
                d["track_confirmed"] = bool(t.confirmed)
            else:
                tid = self.next_id
                self.next_id += 1
                t = Track(tid, d["box"], d["cls"], d["conf"], d["label"])
                self.tracks[tid] = t
                assigned_tracks.add(tid)
                d["track_id"] = int(tid)
                d["track_confirmed"] = bool(t.confirmed)

        dead = [tid for tid, t in self.tracks.items() if t.age > TRACK_MAX_AGE]
        for tid in dead:
            del self.tracks[tid]

        return det_list

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not detected. Try VideoCapture(1).")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")
    h, w = frame0.shape[:2]

    writer = None
    if SAVE_OVERLAY_VIDEO:
        out_path = str(run_dir / "overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, VIDEO_FPS, (w, h))

    jsonl_path = run_dir / "detections.jsonl"
    f = open(jsonl_path, "w", encoding="utf-8")

    meta = {
        "run_dir": str(run_dir),
        "model_path": MODEL_PATH,
        "imgsz": IMGSZ,
        "conf_thres": CONF_THRES,
        "iou_thres": IOU_THRES,
        "max_det": MAX_DET,
        "frame_w": w,
        "frame_h": h,
        "save_overlay_video": SAVE_OVERLAY_VIDEO,
        "video_fps": VIDEO_FPS,
        "tracking": {
            "method": "iou-greedy",
            "track_iou_thres": TRACK_IOU_THRES,
            "track_max_age": TRACK_MAX_AGE,
            "track_min_hits": TRACK_MIN_HITS,
            "smooth_alpha": TRACK_SMOOTH_ALPHA,
            "class_consistent": True
        },
        "started_ts": now_s(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tracker = IoUTracker()

    frame_id = 0
    t0 = now_s()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_id += 1
        ts = now_s()

        inp = preprocess_bgr(frame, IMGSZ)

        t_infer0 = now_s()
        out = sess.run(None, {input_name: inp})[0]
        infer_ms = (now_s() - t_infer0) * 1000.0

        det = postprocess_yolov8(out, conf_thresh=CONF_THRES, iou_thresh=IOU_THRES, max_det=MAX_DET)
        det = scale_boxes(det, IMGSZ, w, h)

        det_list = []
        for x1, y1, x2, y2, conf, cls in det:
            cls = int(cls)
            det_list.append({
                "cls": cls,
                "label": COCO80[cls] if 0 <= cls < len(COCO80) else f"cls{cls}",
                "conf": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

        det_list = tracker.step(det_list)

        fps = frame_id / max(1e-6, (ts - t0))
        for d in det_list:
            x1, y1, x2, y2 = d["box"]
            tid = d.get("track_id", -1)
            conf = d["conf"]
            lbl = d["label"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID{tid} {lbl} {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"FPS {fps:.1f} | infer {infer_ms:.1f}ms | det {len(det_list)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if SAVE_OVERLAY_VIDEO and writer is not None:
            writer.write(frame)

        if frame_id % LOG_EVERY_N_FRAMES == 0:
            rec = {
                "ts": ts,
                "frame": frame_id,
                "infer_ms": infer_ms,
                "fps_est": fps,
                "detections": det_list,
            }
            f.write(json.dumps(rec) + "\n")

        cv2.imshow("MAPLE SHIELD - Detect + Track + Log", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    f.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    meta["ended_ts"] = now_s()
    meta["frames"] = frame_id
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved run to: {run_dir}")
    print(f"JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()

