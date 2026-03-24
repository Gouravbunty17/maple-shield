import json
import argparse
from pathlib import Path
import cv2

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

def get_risk_color(risk_state):
    if risk_state == "UNSAFE":
        return (0, 0, 255)
    elif risk_state == "CAUTION":
        return (0, 165, 255)
    else:
        return (0, 255, 0)

def draw_detection(frame, det, class_names):
    x1, y1, x2, y2 = map(int, det.get("box", [0, 0, 0, 0]))
    conf = float(det.get("conf", 0.0))
    cls = int(det.get("cls", -1))
    class_name = det.get("label")
    
    if not class_name:
        class_name = class_names[cls] if 0 <= cls < len(class_names) else f"cls{cls}"
    
    track_id = det.get("track_id", None)
    
    # Risk info
    risk = det.get("risk", {}) or {}
    risk_state = risk.get("risk_state", "SAFE")
    risk_score = float(risk.get("risk_score", 0.0))
    
    # Velocity + persistence
    vel = det.get("velocity", {}) or {}
    speed = float(vel.get("speed", 0.0))
    persistence = int(det.get("persistence_count", 0))
    
    # Color by risk state
    color = get_risk_color(risk_state)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Build label
    label = f"{class_name} {conf:.2f} | {risk_state} {risk_score:.2f}"
    if speed > 0:
        label += f" | v={speed:.1f}px/f"
    if persistence > 0:
        label += f" | p={persistence}"
    if track_id is not None:
        label = f"ID{int(track_id)} {label}"
    
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y0 = max(0, y1 - th - 6)
    cv2.rectangle(frame, (x1, y0), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--video", type=str, default="")
    args = ap.parse_args()
    
    run_dir = Path(args.run_dir)
    jsonl_path = run_dir / "detections.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing: {jsonl_path}")
    
    # ALWAYS prefer raw.mp4 to avoid double-overlay
    if args.video:
        video_path = run_dir / args.video
    else:
        raw = run_dir / "raw.mp4"
        overlay = run_dir / "overlay.mp4"
        video_path = raw if raw.exists() else overlay
    
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    
    # Load JSONL
    by_frame = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                fr = int(r.get("frame", -1))
                if fr >= 1:
                    by_frame[fr] = r
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = int((1000.0 / fps) / max(0.05, args.speed))
    
    print(f"Run: {run_dir.name}")
    print(f"Source: {video_path.name}")
    print("Keys: SPACE=pause | q=quit\n")
    
    paused = False
    frame_idx = 0
    
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            
            rec = by_frame.get(frame_idx)
            if rec:
                dets = rec.get("detections", [])
                for d in dets:
                    draw_detection(frame, d, COCO80)
                
                # Header with max risk
                max_risk = rec.get("max_risk_state", "SAFE")
                max_score = rec.get("max_risk_score", 0.0)
                infer_ms = float(rec.ggit add .et("infer_ms", 0))
                
                info = f"frame {frame_idx} | {infer_ms:.1f}ms | {len(dets)} det | MAX: {max_risk} {max_score:.2f}"
                color = get_risk_color(max_risk)
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("MAPLE SHIELD Replay", frame)
        k = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

