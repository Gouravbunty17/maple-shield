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
# Config (edit as needed)
# -----------------------
MODEL_PATH = r"models\yolov8n.onnx"
RUNS_DIR = Path("runs")
IMGSZ = 640

CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_DET = 50

SAVE_OVERLAY_VIDEO = True
VIDEO_FPS = 30.0

LOG_EVERY_N_FRAMES = 1       # set to 5 or 10 if you want smaller logs
SAVE_FRAME_HASH = True       # creates a content hash for auditing, cheap-ish

# -----------------------
# Utils
# -----------------------
def now_s():
    return time.time()

def sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def preprocess_bgr(frame, size=640):
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
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
    # out: (1,84,8400) -> pred: (8400,84)
    pred = out[0].transpose(1, 0).astype(np.float32)

    boxes_xywh = pred[:, 0:4]
    cls_scores = pred[:, 4:]  # (8400,80)
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
    return det  # (N,6) x1,y1,x2,y2,conf,cls

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

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    # --- Run directory ---
    run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- ORT session ---
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output0 = sess.get_outputs()[0].name

    # --- Camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not detected. Try VideoCapture(1).")

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")
    h, w = frame0.shape[:2]

    # --- Video writer (overlay) ---
    writer = None
    if SAVE_OVERLAY_VIDEO:
        out_path = str(run_dir / "overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, VIDEO_FPS, (w, h))

    # --- Logs ---
    jsonl_path = run_dir / "detections.jsonl"
    f = open(jsonl_path, "w", encoding="utf-8")

    meta = {
        "run_dir": str(run_dir),
        "model_path": MODEL_PATH,
        "input_name": input_name,
        "output0_name": output0,
        "imgsz": IMGSZ,
        "conf_thres": CONF_THRES,
        "iou_thres": IOU_THRES,
        "max_det": MAX_DET,
        "frame_w": w,
        "frame_h": h,
        "save_overlay_video": SAVE_OVERLAY_VIDEO,
        "video_fps": VIDEO_FPS,
        "log_every_n_frames": LOG_EVERY_N_FRAMES,
        "save_frame_hash": SAVE_FRAME_HASH,
        "started_ts": now_s(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- Loop ---
    frame_id = 0
    t0 = now_s()
    last_print = now_s()

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

        # Build detection records
        det_list = []
        for x1, y1, x2, y2, conf, cls in det:
            cls = int(cls)
            det_list.append({
                "cls": cls,
                "label": COCO80[cls] if 0 <= cls < len(COCO80) else f"cls{cls}",
                "conf": float(conf),
                "box": [int(x1), int(y1), int(x2), int(y2)]
            })

        # HUD / overlay
        fps = frame_id / max(1e-6, (ts - t0))
        for d in det_list:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'{d["label"]} {d["conf"]:.2f}', (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"FPS {fps:.1f} | infer {infer_ms:.1f}ms | det {len(det_list)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if SAVE_OVERLAY_VIDEO and writer is not None:
            writer.write(frame)

        # Logging (JSONL)
        if frame_id % LOG_EVERY_N_FRAMES == 0:
            rec = {
                "ts": ts,
                "frame": frame_id,
                "infer_ms": infer_ms,
                "fps_est": fps,
                "detections": det_list,
            }
            if SAVE_FRAME_HASH:
                # hash the raw frame bytes (BGR) - useful for audit/replay integrity checks
                rec["frame_sha1"] = sha1_of_bytes(frame.tobytes())
            f.write(json.dumps(rec) + "\n")

        # Print metrics
        if ts - last_print > 1.0:
            print(f"[metrics] frame={frame_id} fps={fps:.1f} infer_ms={infer_ms:.1f} det={len(det_list)}")
            last_print = ts

        cv2.imshow("MAPLE SHIELD - Detect + Logging", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    f.close()
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    # Final meta update
    meta["ended_ts"] = now_s()
    meta["frames"] = frame_id
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved run to: {run_dir}")
    print(f"JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()

