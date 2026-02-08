import json
import time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from borealis_risk import compute_risk_score, get_risk_color
from borealis_tracker_v2 import IoUTracker

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

MODEL_PATH = r"models\yolov8n.onnx"
RUNS_DIR = Path("runs")
IMGSZ = 640
CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_DET = 50
VIDEO_FPS = 30.0
LOG_EVERY_N_FRAMES = 1

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
    det = np.stack([boxes_xyxy[keep, 0], boxes_xyxy[keep, 1], boxes_xyxy[keep, 2], boxes_xyxy[keep, 3], conf[keep], cls_id[keep].astype(np.float32)], axis=1)
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

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not detected.")
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Camera opened but cannot read frames.")
    h, w = frame0.shape[:2]
    
    # TWO writers: raw (source of truth) + overlay (disposable UI)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    raw_writer = cv2.VideoWriter(str(run_dir / "raw.mp4"), fourcc, VIDEO_FPS, (w, h))
    overlay_writer = cv2.VideoWriter(str(run_dir / "overlay.mp4"), fourcc, VIDEO_FPS, (w, h))
    
    jsonl_path = run_dir / "detections.jsonl"
    f = open(jsonl_path, "w", encoding="utf-8")
    meta = {"run_dir": str(run_dir), "model_path": MODEL_PATH, "imgsz": IMGSZ, "started_ts": now_s()}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tracker = IoUTracker()
    frame_id = 0
    t0 = now_s()
    
    print(f"🎥 Recording to: {run_dir}")
    print("📊 Velocity + lifecycle + risk enabled")
    print("💾 Saving raw.mp4 (source) + overlay.mp4 (UI)")
    print("Press 'q' to stop\n")
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        ts = now_s()
        
        # Save raw frame BEFORE any drawing
        raw_writer.write(frame.copy())
        
        inp = preprocess_bgr(frame, IMGSZ)
        t_infer0 = now_s()
        out = sess.run(None, {input_name: inp})[0]
        infer_ms = (now_s() - t_infer0) * 1000.0
        det = postprocess_yolov8(out, conf_thresh=CONF_THRES, iou_thresh=IOU_THRES, max_det=MAX_DET)
        det = scale_boxes(det, IMGSZ, w, h)
        det_list = []
        for x1, y1, x2, y2, conf, cls in det:
            cls = int(cls)
            det_list.append({"cls": cls, "label": COCO80[cls] if 0 <= cls < len(COCO80) else f"cls{cls}", "conf": float(conf), "box": [int(x1), int(y1), int(x2), int(y2)]})
        det_list = tracker.step(det_list, frame_id, ts)
        
        for d in det_list:
            risk_data = compute_risk_score(d["box"], d["conf"], w, h)
            d["risk"] = risk_data
        
        max_risk = 0.0
        max_risk_state = "SAFE"
        if det_list:
            max_risk = max(d["risk"]["risk_score"] for d in det_list)
            max_risk_state = next(d["risk"]["risk_state"] for d in det_list if d["risk"]["risk_score"] == max_risk)
        
        fps = frame_id / max(1e-6, (ts - t0))
        
        # Draw overlay on COPY of frame
        overlay_frame = frame.copy()
        for d in det_list:
            x1, y1, x2, y2 = d["box"]
            tid = d.get("track_id", -1)
            conf = d["conf"]
            lbl = d["label"]
            risk_state = d["risk"]["risk_state"]
            risk_score = d["risk"]["risk_score"]
            vel = d.get("velocity", {})
            speed = vel.get("speed", 0.0)
            
            color = get_risk_color(risk_state)
            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"ID{tid} {lbl} {conf:.2f} | {risk_state} {risk_score:.2f} | v={speed:.1f}px/f"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay_frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(overlay_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        
        header = f"FPS {fps:.1f} | {infer_ms:.1f}ms | {len(det_list)} det | MAX RISK: {max_risk_state} {max_risk:.2f}"
        header_color = get_risk_color(max_risk_state)
        cv2.putText(overlay_frame, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
        
        # Save overlay frame
        overlay_writer.write(overlay_frame)
        
        if frame_id % LOG_EVERY_N_FRAMES == 0:
            rec = {"ts": ts, "frame": frame_id, "infer_ms": infer_ms, "fps_est": fps, "detections": det_list, "max_risk_score": max_risk, "max_risk_state": max_risk_state}
            f.write(json.dumps(rec) + "\n")
        
        cv2.imshow("BOREALIS V1 - Velocity + Lifecycle", overlay_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    f.close()
    raw_writer.release()
    overlay_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    lifecycle_summary = tracker.get_lifecycle_summary()
    (run_dir / "lifecycle.json").write_text(json.dumps(lifecycle_summary, indent=2), encoding="utf-8")
    
    meta["ended_ts"] = now_s()
    meta["frames"] = frame_id
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    print(f"\n✅ Run complete: {run_dir.name}")
    print(f"📊 Frames logged: {frame_id}")
    print(f"🔄 Total tracks created: {lifecycle_summary['total_tracks_created']}")
    print(f"💀 Tracks ended: {lifecycle_summary['dead_tracks']}")

if __name__ == "__main__":
    main()
